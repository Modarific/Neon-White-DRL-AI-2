#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imitation_video_pretrain.py

Pretrain a low-level "human intention" prior from raw gameplay videos.

Overview
--------
We learn:
  1) Behavior VAE: q(z|a) and decoder p(a|z) over short clips, where "a"
     are vision-derived action proxies (global optical-flow stats).
  2) State-conditioned prior: p(z|s) predicts z from state s (frame embedding + flow summary).

Outputs
-------
- out_dir/checkpoints/{run_name}_step_*.pt
- out_dir/final/{run_name}_behavior_prior.pt
- out_dir/z_bank/{run_name}_clips_z.npz
- out_dir/tensorboard/{run_name}/...

Example (PowerShell)
--------------------
python -m python.imitation_video_pretrain ^
  --video-root "D:/NeonWhiteVideos" ^
  --out-dir   "D:/NW_IL_OUT" ^
  --run-name  "nw_il_dml" ^
  --clip-len  12 --fps 12 --frame-size 160 ^
  --batch-size 8 --num-workers 0 --epochs 10 ^
  --use-pretrained true --freeze-backbone true ^
  --mask-top-ratio 0.12 ^
  --device dml --no-tensorboard --opt adam --no-foreach

Notes
-----
- Farneback optical flow (OpenCV).
- HUD-mask defaults: top 12%.
- Flags: --dry-run, --no-tensorboard, --print-every, --skip-bad-clips,
         --device {auto,cpu,cuda,dml}, --opt, --no-foreach,
         --prefetch-factor, --ram-cache {off,uint8,float32}, --ram-cache-clips.
"""

from __future__ import annotations
import argparse, random, time, os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
from collections import OrderedDict

# Keep TF quiet if tensorboard is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional AMD DirectML
try:
    import torch_directml as _dml
except Exception:
    _dml = None

try:
    import torchvision
    from torchvision import transforms
except ImportError as e:
    raise RuntimeError("Please install torchvision: pip install torchvision") from e


# ------------------------------- Utils ----------------------------------------

def _cv_worker_init(_):
    try:
        cv2.setNumThreads(0)   # one thread per worker, no oversubscription
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_videos(root: Path, exts=(".mp4", ".mkv", ".avi", ".mov")) -> List[Path]:
    root = Path(root)
    vids = []
    for ext in exts:
        vids.extend(root.rglob(f"*{ext}"))
    return sorted(set(vids))


def center_crop_resize(img: np.ndarray, out_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty frame read from video.")
    if h > w:
        top = (h - w) // 2
        img = img[top:top + w, :]
    elif w > h:
        left = (w - h) // 2
        img = img[:, left:left + h]
    img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return img


def apply_hud_mask(img: np.ndarray, top=0.12, bottom=0.0, left=0.0, right=0.0) -> np.ndarray:
    h, w = img.shape[:2]
    t = int(h * float(top)); b = h - int(h * float(bottom))
    l = int(w * float(left)); r = w - int(w * float(right))
    if t > 0: img[:t, :, :] = 0
    if bottom > 0: img[b:, :, :] = 0
    if left > 0: img[:, :l, :] = 0
    if right > 0: img[:, r:, :] = 0
    return img


def safe_optical_flow(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray, next=gray,
        flow=None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow.astype(np.float32)


def flow_to_proxy(flow: np.ndarray) -> np.ndarray:
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)
    mag = np.sqrt(u*u + v*v) + 1e-8
    ang = np.arctan2(v, u)
    proxy = np.array([
        float(u.mean()),
        float(v.mean()),
        float(mag.mean()),
        float(mag.std()),
        float(np.cos(ang).mean()),
        float(np.sin(ang).mean()),
    ], dtype=np.float32)
    return proxy


# ------------------------------ Dataset ---------------------------------------

@dataclass
class ClipIndex:
    video_path: Path
    start_frame: int
    native_fps: float
    native_len: int


class VideoClipDataset(Dataset):
    def __init__(
        self,
        video_root: str | Path,
        clip_len: int = 12,
        fps: int = 12,
        frame_size: int = 160,
        stride: int = 1,
        max_videos: Optional[int] = None,
        subsample_stride: int = 1,
        extensions: Tuple[str, ...] = (".mp4", ".mkv", ".avi", ".mov"),
        mask_top_ratio: float = 0.12,
        mask_bottom_ratio: float = 0.0,
        mask_left_ratio: float = 0.0,
        mask_right_ratio: float = 0.0,
        skip_bad_clips: bool = False,
        max_resample_tries: int = 5,
        # RAM cache
        ram_cache: str = "off",              # "off" | "uint8" | "float32"
        ram_cache_clips: int = 0,            # 0 = unlimited (be careful)
    ):
        super().__init__()
        self.clip_len = int(clip_len)
        self.fps = int(fps)
        self.frame_size = int(frame_size)
        self.stride = max(1, int(stride))
        self.subsample_stride = max(1, int(subsample_stride))
        self.videos = list_videos(Path(video_root), exts=extensions)
        if max_videos is not None:
            self.videos = self.videos[:max_videos]
        if len(self.videos) == 0:
            raise RuntimeError(f"No videos found under {video_root}")

        def _clamp01(x: float) -> float:
            return float(max(0.0, min(1.0, x)))
        self.mask_top_ratio = _clamp01(mask_top_ratio)
        self.mask_bottom_ratio = _clamp01(mask_bottom_ratio)
        self.mask_left_ratio = _clamp01(mask_left_ratio)
        self.mask_right_ratio = _clamp01(mask_right_ratio)

        self.skip_bad_clips = bool(skip_bad_clips)
        self.max_resample_tries = int(max_resample_tries)

        # RAM cache
        self.ram_cache_mode = ram_cache
        self.ram_cache_limit = int(ram_cache_clips)
        self._ram_cache: "OrderedDict[tuple, tuple]" = OrderedDict()

        self.indices: List[ClipIndex] = []
        self._build_index()

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ----- RAM cache helpers -----

    def _cache_key(self, idx: ClipIndex) -> tuple:
        return (str(idx.video_path), idx.start_frame, self.frame_size, self.clip_len, self.fps,
                self.mask_top_ratio, self.mask_bottom_ratio, self.mask_left_ratio, self.mask_right_ratio)

    def _cache_get(self, key: tuple):
        if self.ram_cache_mode == "off":
            return None
        val = self._ram_cache.get(key)
        if val is not None:
            self._ram_cache.move_to_end(key, last=True)
        return val

    def _cache_put(self, key: tuple, frames_rgb: List[np.ndarray], proxies_np: np.ndarray):
        if self.ram_cache_mode == "off":
            return
        # LRU trim if needed
        if self.ram_cache_limit > 0:
            while len(self._ram_cache) >= self.ram_cache_limit:
                self._ram_cache.popitem(last=False)
        if self.ram_cache_mode == "uint8":
            frames_pack = [f.copy() for f in frames_rgb]  # keep raw uint8 RGB
        else:  # "float32": store normalized tensor to skip transforms later
            frames_pack = torch.stack([self.to_tensor(f) for f in frames_rgb], dim=0)  # (T,3,H,W) float32
        self._ram_cache[key] = (frames_pack, proxies_np.copy())
        self._ram_cache.move_to_end(key, last=True)

    # ----- indexing / reading -----

    def _build_index(self) -> None:
        for vp in self.videos:
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                continue
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            native_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if native_len <= 0:
                continue

            step = max(1, int(round(native_fps / self.fps)))
            clip_native = (self.clip_len - 1) * step + 1

            start = 0
            while start + clip_native <= native_len:
                self.indices.append(ClipIndex(vp, start, native_fps, native_len))
                start += self.stride * step

        if len(self.indices) == 0:
            raise RuntimeError("Could not build any clips; check fps/clip_len vs videos.")

    def __len__(self) -> int:
        return len(self.indices)

    def _read_clip_frames(self, idx: ClipIndex) -> Tuple[List[np.ndarray], int]:
        cap = cv2.VideoCapture(str(idx.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {idx.video_path}")
        native_fps = idx.native_fps or 30.0
        step = max(1, int(round(native_fps / self.fps)))

        frames: List[np.ndarray] = []
        target_positions = [idx.start_frame + i * step for i in range(self.clip_len)]
        for pos in target_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame = center_crop_resize(frame, self.frame_size)
            frame = apply_hud_mask(
                frame,
                top=self.mask_top_ratio,
                bottom=self.mask_bottom_ratio,
                left=self.mask_left_ratio,
                right=self.mask_right_ratio,
            )
            frames.append(frame)
        cap.release()
        if len(frames) != self.clip_len:
            raise RuntimeError(f"Short read in {idx.video_path} starting {idx.start_frame}")
        return frames, step

    def _getitem_once(self, i: int):
        idx = self.indices[i]
        key = self._cache_key(idx)

        # Try RAM cache
        cached = self._cache_get(key)
        if cached is not None:
            frames_pack, proxies_np = cached
            if self.ram_cache_mode == "uint8":
                frames_tensor = torch.stack([self.to_tensor(f) for f in frames_pack], dim=0)
            else:  # float32 tensor already normalized
                frames_tensor = frames_pack.clone()
            proxies_tensor = torch.from_numpy(proxies_np.copy())
            state_aux = torch.from_numpy(proxies_np.mean(axis=0).astype(np.float32))
            return {
                "frames": frames_tensor,
                "proxies": proxies_tensor,
                "state_aux": state_aux,
                "meta": {
                    "video": str(idx.video_path),
                    "start_frame": idx.start_frame,
                    "step_native": None,
                }
            }

        # Decode & compute flow/proxies
        frames_rgb, step = self._read_clip_frames(idx)
        gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]
        flows = [safe_optical_flow(gray[t-1], gray[t]) for t in range(1, len(gray))]
        proxies_np = np.stack([flow_to_proxy(f) for f in flows], axis=0).astype(np.float32)  # (T-1,6)
        self._cache_put(key, frames_rgb, proxies_np)

        frames_tensor = torch.stack([self.to_tensor(f) for f in frames_rgb], dim=0)  # (T,3,H,W)
        proxies_tensor = torch.from_numpy(proxies_np)
        state_aux = torch.from_numpy(proxies_np.mean(axis=0).astype(np.float32))

        return {
            "frames": frames_tensor,
            "proxies": proxies_tensor,
            "state_aux": state_aux,
            "meta": {
                "video": str(idx.video_path),
                "start_frame": idx.start_frame,
                "step_native": step,
            }
        }

    def __getitem__(self, i: int):
        if not self.skip_bad_clips:
            return self._getitem_once(i)

        tries = 0
        last_err: Optional[Exception] = None
        while tries < self.max_resample_tries:
            try:
                return self._getitem_once(i if tries == 0 else random.randrange(len(self.indices)))
            except Exception as e:
                last_err = e
                tries += 1
        raise last_err if last_err is not None else RuntimeError("Unknown clip error")


# ----------------------------- Models (DML-safe GRU) ---------------------------

class SafeGRU(nn.Module):
    """
    Unfused single-layer GRU (batch_first) via Linear ops.
    Works on CPU/CUDA/DirectML (avoids fused kernels).
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x_proj = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.h_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        # x: (B, T, I)
        B, T, _ = x.shape
        H = self.hidden_size
        h = x.new_zeros((B, H)) if h0 is None else h0
        outputs = []
        for t in range(T):
            x_lin = self.x_proj(x[:, t, :])      # (B, 3H)
            h_lin = self.h_proj(h)               # (B, 3H)
            x_z, x_r, x_n = x_lin.chunk(3, dim=1)
            h_z, h_r, h_n = h_lin.chunk(3, dim=1)
            z = torch.sigmoid(x_z + h_z)
            r = torch.sigmoid(x_r + h_r)
            n = torch.tanh(x_n + r * h_n)
            h = (1 - z) * n + z * h
            outputs.append(h.unsqueeze(1))
        y = torch.cat(outputs, dim=1)  # (B,T,H)
        return y, h


class FrameEncoder(nn.Module):
    """CNN backbone to encode a single RGB frame (default: ResNet-18 -> 512-d)."""
    def __init__(self, name="resnet18", pretrained=True, train_backbone=False, out_dim=512):
        super().__init__()
        if name != "resnet18":
            raise NotImplementedError("Only resnet18 is wired by default.")
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            net = torchvision.models.resnet18(weights=weights)
        except Exception:
            net = torchvision.models.resnet18(pretrained=pretrained)
        modules = list(net.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.out_dim = out_dim
        self.fc = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad = bool(train_backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)          # (B,512,1,1)
        f = torch.flatten(f, 1)       # (B,512)
        return self.fc(f)             # (B,512)


class BehaviorEncoder(nn.Module):
    """q(z|a): Encode proxies -> latent Gaussian (mu, logvar) using SafeGRU."""
    def __init__(self, a_dim=6, z_dim=32, hidden=256):
        super().__init__()
        self.gru = SafeGRU(input_size=a_dim, hidden_size=hidden)
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, a_seq: torch.Tensor):
        _, h = self.gru(a_seq)   # (B, hidden)
        return self.mu(h), self.logvar(h)


class BehaviorDecoder(nn.Module):
    """p(a|z): Reconstruct proxy sequence using SafeGRU."""
    def __init__(self, a_dim=6, z_dim=32, hidden=256, Tm1=11):
        super().__init__()
        self.Tm1 = Tm1
        self.fc_in = nn.Linear(z_dim, hidden)   # repeated per-step input
        self.fc_h0 = nn.Linear(z_dim, hidden)   # initial hidden
        self.gru = SafeGRU(input_size=hidden, hidden_size=hidden)
        self.head = nn.Linear(hidden, a_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        inp = torch.tanh(self.fc_in(z)).unsqueeze(1).repeat(1, self.Tm1, 1)  # (B,T-1,H)
        h0 = torch.tanh(self.fc_h0(z))                                       # (B,H)
        y, _ = self.gru(inp, h0=h0)                                          # (B,T-1,H)
        a_hat = self.head(y)                                                 # (B,T-1,a_dim)
        return a_hat


class PriorNet(nn.Module):
    """p(z|s): Predict latent intention from state s = [frame_feat, flow_summary]."""
    def __init__(self, frame_feat_dim=512, flow_dim=6, z_dim=32, hidden=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(frame_feat_dim + flow_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, frame_feat: torch.Tensor, flow_summary: torch.Tensor):
        x = torch.cat([frame_feat, flow_summary], dim=1)
        h = self.fc(x)
        return self.mu(h), self.logvar(h)


# ----------------------------- Loss helpers -----------------------------------

def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, dim=1)


def kl_gaussians(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                 mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    term = (logvar_p - logvar_q) + (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8) - 1.0
    return 0.5 * torch.sum(term, dim=1)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# --------------------------- Device & Optimizer --------------------------------

def pick_device(args):
    if args.device == "cpu":
        return torch.device("cpu"), "cpu"
    if args.device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        print("[device] WARN: --device cuda requested but CUDA not available; falling back to CPU.")
        return torch.device("cpu"), "cpu"
    if args.device == "dml":
        if _dml is None:
            raise RuntimeError("torch-directml not installed (pip install torch-directml).")
        return _dml.device(), "dml"
    # auto
    if _dml is not None:
        return _dml.device(), "dml"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def build_optimizer(params, args, dev_kind: str):
    use_foreach = (not args.no_foreach) and (dev_kind != "dml")
    kwargs = dict(lr=args.lr)
    if args.opt in ("adam", "adamw"):
        kwargs["foreach"] = use_foreach

    if args.opt == "adam":
        opt_cls = torch.optim.Adam
    elif args.opt == "adamw":
        opt_cls = torch.optim.AdamW
    elif args.opt == "sgd":
        opt_cls = torch.optim.SGD
        kwargs.update(momentum=0.9, nesterov=True)
    elif args.opt == "rmsprop":
        opt_cls = torch.optim.RMSprop
        kwargs.update(momentum=0.9, alpha=0.99)
    else:
        raise ValueError(f"Unknown optimizer {args.opt}")

    try:
        opt = opt_cls(params, **kwargs)
    except TypeError:
        kwargs.pop("foreach", None)
        opt = opt_cls(params, **kwargs)

    print(f"[opt] {args.opt} foreach={kwargs.get('foreach', False)}")
    if dev_kind == "dml" and (args.opt in ("adam", "adamw")) and (not use_foreach):
        print("[device] DirectML detected; foreach disabled to avoid CPU fallbacks.")
    return opt


# --------------------------- Training / Export --------------------------------

@dataclass
class TrainConfig:
    clip_len: int = 12
    fps: int = 12
    frame_size: int = 160
    batch_size: int = 12
    epochs: int = 10
    lr: float = 2e-4
    z_dim: int = 32
    hidden: int = 256
    beta_kl: float = 1e-3
    lambda_prior: float = 1.0
    log_interval: int = 50
    ckpt_interval: int = 1000
    use_pretrained: bool = True
    freeze_backbone: bool = True


def train_loop(args: argparse.Namespace) -> None:
    device, dev_kind = pick_device(args)
    set_seed(args.seed)
    print(f"[device] Using {dev_kind.upper()} -> {device}")

    # IO
    out_dir = Path(args.out_dir).expanduser().resolve()
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "final").mkdir(parents=True, exist_ok=True)
    (out_dir / "z_bank").mkdir(parents=True, exist_ok=True)
    tb_dir = out_dir / "tensorboard" / args.run_name

    writer = None
    if (not args.no_tensorboard) and args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(tb_dir))

    # Data
    ds = VideoClipDataset(
        video_root=args.video_root,
        clip_len=args.clip_len,
        fps=args.fps,
        frame_size=args.frame_size,
        stride=args.stride,
        max_videos=args.max_videos,
        mask_top_ratio=args.mask_top_ratio,
        mask_bottom_ratio=args.mask_bottom_ratio,
        mask_left_ratio=args.mask_left_ratio,
        mask_right_ratio=args.mask_right_ratio,
        skip_bad_clips=args.skip_bad_clips,
        max_resample_tries=args.max_resample_tries,
        ram_cache=args.ram_cache,
        ram_cache_clips=args.ram_cache_clips,
    )
    pin_mem = (dev_kind == "cuda")
    dl_kwargs = dict(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = max(2, args.prefetch_factor)
        dl_kwargs["worker_init_fn"] = _cv_worker_init
    dl = DataLoader(**dl_kwargs)

    # Dry-run
    if args.dry_run:
        print(f"[dry-run] videos={len(ds.videos)} clips={len(ds)} "
              f"(clip_len={args.clip_len}, fps={args.fps}, frame_size={args.frame_size})")
        sample = next(iter(dl))
        f, p = sample["frames"], sample["proxies"]
        print(f"[dry-run] batch frames={tuple(f.shape)} proxies={tuple(p.shape)}")
        if writer:
            writer.flush(); writer.close()
        return

    # Models
    frame_enc = FrameEncoder(pretrained=args.use_pretrained,
                             train_backbone=not args.freeze_backbone).to(device)
    beh_enc = BehaviorEncoder(a_dim=6, z_dim=args.z_dim, hidden=args.hidden).to(device)
    beh_dec = BehaviorDecoder(a_dim=6, z_dim=args.z_dim, hidden=args.hidden, Tm1=args.clip_len-1).to(device)
    prior = PriorNet(frame_feat_dim=512, flow_dim=6, z_dim=args.z_dim, hidden=args.hidden).to(device)

    params = list(beh_enc.parameters()) + list(beh_dec.parameters()) + list(prior.parameters())
    if not args.freeze_backbone:
        params += list(frame_enc.parameters())

    opt = build_optimizer(params, args, dev_kind)

    global_step = 0
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        frame_enc.train(); beh_enc.train(); beh_dec.train(); prior.train()
        for batch in dl:
            frames = batch["frames"].to(device)
            proxies = batch["proxies"].to(device)
            flow_sum = batch["state_aux"].to(device)

            f0 = frames[:, 0]
            with torch.set_grad_enabled(not args.freeze_backbone):
                frame_feat = frame_enc(f0)

            mu_q, logvar_q = beh_enc(proxies)
            z = reparameterize(mu_q, logvar_q)

            a_hat = beh_dec(z)
            mu_p, logvar_p = prior(frame_feat, flow_sum)

            rec = F.mse_loss(a_hat, proxies, reduction="none").mean(dim=(1,2))
            kl_std = kl_standard_normal(mu_q, logvar_q)
            kl_prior = kl_gaussians(mu_q, logvar_q, mu_p, logvar_p)

            loss = rec + args.beta_kl * kl_std + args.lambda_prior * kl_prior
            loss_mean = loss.mean()

            opt.zero_grad(set_to_none=True)
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()

            if (args.print_every > 0) and (global_step % args.print_every == 0):
                print(f"[step {global_step}] loss={loss_mean.item():.4f} "
                      f"rec={rec.mean().item():.4f} "
                      f"kl_std={kl_std.mean().item():.4f} "
                      f"kl_prior={kl_prior.mean().item():.4f}")

            if writer and (global_step % args.log_interval == 0):
                with torch.no_grad():
                    writer.add_scalar("train/loss_total", loss_mean.item(), global_step)
                    writer.add_scalar("train/rec", rec.mean().item(), global_step)
                    writer.add_scalar("train/kl_std", kl_std.mean().item(), global_step)
                    writer.add_scalar("train/kl_prior", kl_prior.mean().item(), global_step)
                    writer.add_scalar("meta/fps", global_step / max(1e-6, (time.time() - start_time)), global_step)

            if (global_step % args.ckpt_interval) == 0 and global_step > 0:
                ckpt_path = (out_dir / "checkpoints" / f"{args.run_name}_step_{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "frame_enc": frame_enc.state_dict(),
                    "beh_enc": beh_enc.state_dict(),
                    "beh_dec": beh_dec.state_dict(),
                    "prior": prior.state_dict(),
                    "cfg": vars(args),
                }, ckpt_path)

            best_loss = min(best_loss, loss_mean.item())
            global_step += 1

        print(f"[epoch {epoch}] loss={loss_mean.item():.4f} best={best_loss:.4f}")

    # Save final bundle
    final_path = out_dir / "final" / f"{args.run_name}_behavior_prior.pt"
    torch.save({
        "frame_enc": frame_enc.state_dict(),
        "beh_enc": beh_enc.state_dict(),
        "beh_dec": beh_dec.state_dict(),
        "prior": prior.state_dict(),
        "z_dim": args.z_dim,
        "frame_feat_dim": 512,
        "a_dim": 6,
        "clip_len": args.clip_len,
        "cfg": vars(args),
    }, final_path)
    print(f"[final] saved to {final_path}")

    if writer:
        writer.flush(); writer.close()

    if args.export_z_bank:
        export_z_bank(args, ds, frame_enc, beh_enc, prior, device, out_dir)


@torch.no_grad()
def export_z_bank(
    args: argparse.Namespace,
    ds: VideoClipDataset,
    frame_enc: FrameEncoder,
    beh_enc: BehaviorEncoder,
    prior: PriorNet,
    device: torch.device,
    out_dir: Path,
) -> None:
    frame_enc.eval(); beh_enc.eval(); prior.eval()
    z_list = []
    meta_list = []

    # pin_memory only helps CUDA
    dl = DataLoader(ds, batch_size=args.zbank_batch, shuffle=False,
                    num_workers=args.num_workers, pin_memory=False, drop_last=False)
    for batch in dl:
        frames = batch["frames"].to(device)
        proxies = batch["proxies"].to(device)
        flow_sum = batch["state_aux"].to(device)

        f0 = frames[:, 0]
        frame_feat = frame_enc(f0)

        mu_q, _ = beh_enc(proxies)
        z = mu_q

        z_list.append(z.cpu().numpy())
        vids = batch["meta"]["video"]
        starts = batch["meta"]["start_frame"]
        for i in range(len(vids)):
            meta_list.append(f"{vids[i]}#{starts[i]}")

    z_all = np.concatenate(z_list, axis=0)
    out_npz = out_dir / "z_bank" / f"{args.run_name}_clips_z.npz"
    np.savez_compressed(out_npz, z=z_all, meta=np.array(meta_list))
    print(f"[z-bank] {z_all.shape[0]} clip latents saved to {out_npz}")


# --------------------------------- CLI ----------------------------------------

def _bool(s: str) -> bool:
    return str(s).lower() in {"1","true","t","yes","y"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain human-like behavior prior from gameplay videos")

    # IO
    p.add_argument("--video-root", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--run-name", type=str, default=f"il_{int(time.time())}")
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--tensorboard", type=_bool, default=True)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--print-every", type=int, default=50)

    # Data
    p.add_argument("--clip-len", type=int, default=12)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--frame-size", type=int, default=160)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--mask-top-ratio", type=float, default=0.12)
    p.add_argument("--mask-bottom-ratio", type=float, default=0.0)
    p.add_argument("--mask-left-ratio", type=float, default=0.0)
    p.add_argument("--mask-right-ratio", type=float, default=0.0)
    p.add_argument("--skip-bad-clips", action="store_true")
    p.add_argument("--max-resample-tries", type=int, default=5)

    # RAM cache / prefetch
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="Batches prefetched per worker (>=2 when num_workers>0).")
    p.add_argument("--ram-cache", choices=["off","uint8","float32"], default="off",
                   help="Cache decoded clips in RAM: raw RGB uint8 (compact) or normalized float32 tensors.")
    p.add_argument("--ram-cache-clips", type=int, default=0,
                   help="Max clips to keep in RAM cache (0 means unlimited).")

    # Train
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)

    # Device
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "dml"], default="auto",
                   help="Use 'dml' for AMD via DirectML; 'auto' prefers DML, then CUDA, else CPU.")

    # Model
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--beta-kl", type=float, default=1e-3)
    p.add_argument("--lambda-prior", type=float, default=1.0)
    p.add_argument("--use-pretrained", type=_bool, default=True)
    p.add_argument("--freeze-backbone", type=_bool, default=True)

    # Logging/ckpt
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--ckpt-interval", type=int, default=1000)

    # Z-bank export
    p.add_argument("--export-z-bank", action="store_true")
    p.add_argument("--zbank-batch", type=int, default=32)

    # Optimizer
    p.add_argument("--opt", choices=["adam","adamw","sgd","rmsprop"], default="adam",
                   help="Optimizer (default: adam).")
    p.add_argument("--no-foreach", action="store_true",
                   help="Disable foreach optimizer kernels (recommended on DirectML).")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_loop(args)
