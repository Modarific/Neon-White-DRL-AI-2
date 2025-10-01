from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal


@dataclass
class PPOConfig:
    total_steps: int = 500_000
    rollout_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.015


def atanh(x: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)
    return 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))


class TanhNormal:
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        self.mean = mean
        self.log_std = log_std
        self.std = torch.exp(log_std)
        self.normal = Normal(mean, self.std)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.normal.rsample()
        action = torch.tanh(raw)
        log_prob = self.log_prob(action, raw)
        return action, log_prob

    def log_prob(self, action: torch.Tensor, raw: torch.Tensor | None = None) -> torch.Tensor:
        action = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
        if raw is None:
            raw = atanh(action)
        log_prob = self.normal.log_prob(raw) - torch.log(torch.clamp(1 - action.pow(2), min=1e-6))
        return log_prob.sum(-1)

    def entropy(self) -> torch.Tensor:
        return self.normal.entropy()


def orthogonal_init(module: nn.Module) -> None:
    """Apply orthogonal init to Linear layers with tanh gain and zero bias."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(module.bias)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        continuous_dim: int,
        discrete_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256, 256),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last_size = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.Tanh())
            last_size = size
        self.backbone = nn.Sequential(*layers)

        self.actor_mean = nn.Linear(last_size, continuous_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(continuous_dim))
        self.actor_disc = nn.Linear(last_size, discrete_dim)
        self.value_head = nn.Linear(last_size, 1)

        # Orthogonal initialization for stability
        self.backbone.apply(orthogonal_init)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.actor_disc.weight, gain=0.01)
        nn.init.zeros_(self.actor_disc.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def get_dists(self, obs: torch.Tensor) -> Tuple[TanhNormal, Bernoulli, torch.Tensor]:
        features = self.forward(obs)
        mean = self.actor_mean(features)
        log_std = torch.clamp(self.actor_log_std, -20.0, 2.0)
        log_std = log_std.expand_as(mean)
        cont_dist = TanhNormal(mean, log_std)
        disc_logits = self.actor_disc(features)
        disc_dist = Bernoulli(logits=disc_logits)
        value = self.value_head(features).squeeze(-1)
        return cont_dist, disc_dist, value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cont_dist, disc_dist, value = self.get_dists(obs)
        cont_action, cont_log_prob = cont_dist.sample()
        disc_action = disc_dist.sample()
        disc_log_prob = disc_dist.log_prob(disc_action).sum(-1)
        log_prob = cont_log_prob + disc_log_prob
        entropy = cont_dist.entropy().sum(-1) + disc_dist.entropy().sum(-1)
        return cont_action, disc_action, log_prob, value, entropy

    def act_deterministic(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Greedy action: tanh(mean) for continuous, 0.5 threshold for discrete."""
        cont_dist, disc_dist, value = self.get_dists(obs)
        cont_action = torch.tanh(cont_dist.mean)
        disc_action = (disc_dist.probs >= 0.5).float()
        log_prob_cont = cont_dist.log_prob(cont_action)
        log_prob_disc = disc_dist.log_prob(disc_action).sum(-1)
        log_prob = log_prob_cont + log_prob_disc
        entropy = cont_dist.entropy().sum(-1) + disc_dist.entropy().sum(-1)
        return cont_action, disc_action, log_prob, value, entropy

    def evaluate_actions(
        self, obs: torch.Tensor, cont_action: torch.Tensor, disc_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cont_dist, disc_dist, value = self.get_dists(obs)
        log_prob_cont = cont_dist.log_prob(cont_action)
        log_prob_disc = disc_dist.log_prob(disc_action).sum(-1)
        log_prob = log_prob_cont + log_prob_disc
        entropy = cont_dist.entropy().sum(-1) + disc_dist.entropy().sum(-1)
        return log_prob, entropy, value, log_prob_cont


class RolloutBuffer:
    def __init__(
        self,
        num_steps: int,
        obs_dim: int,
        cont_dim: int,
        disc_dim: int,
        device: torch.device,
    ) -> None:
        self.device = device
        self.num_steps = num_steps
        self.obs = torch.zeros((num_steps, obs_dim), dtype=torch.float32, device=device)
        self.cont_actions = torch.zeros((num_steps, cont_dim), dtype=torch.float32, device=device)
        self.disc_actions = torch.zeros((num_steps, disc_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.values = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.dones = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.returns = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        cont_action: torch.Tensor,
        disc_action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: float,
    ) -> None:
        if self.ptr >= self.num_steps:
            raise RuntimeError('Rollout buffer overflow')
        self.obs[self.ptr].copy_(obs)
        self.cont_actions[self.ptr].copy_(cont_action)
        self.disc_actions[self.ptr].copy_(disc_action)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor, last_done: float, gamma: float, gae_lambda: float) -> None:
        last_advantage = 0.0
        last_value_scalar = last_value.item() if isinstance(last_value, torch.Tensor) else float(last_value)
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value_scalar
            else:
                next_non_terminal = 1.0 - self.dones[step + 1].item()
                next_value = self.values[step + 1].item()
            delta = self.rewards[step].item() + gamma * next_value * next_non_terminal - self.values[step].item()
            last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            self.advantages[step] = last_advantage
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def iter_minibatches(self, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        total = self.ptr
        indices = torch.randperm(total, device=self.device)
        for start in range(0, total, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield {
                'obs': self.obs[batch_idx],
                'cont_actions': self.cont_actions[batch_idx],
                'disc_actions': self.disc_actions[batch_idx],
                'log_probs': self.log_probs[batch_idx],
                'advantages': self.advantages[batch_idx],
                'returns': self.returns[batch_idx],
                'values': self.values[batch_idx],
            }

    def reset(self) -> None:
        self.ptr = 0
