// NeonRLBridge.cs
// Build as a MelonLoader mod. Tested with Unity 2019+ APIs.
// Requires: MelonLoader, UnityEngine.
// Place the compiled DLL in Neon White/Mods, run the game, then connect from Python.
//
// New in this version:
// - Enemy discovery and telemetry:
//    * Detects enemies by tag/layer/name/component-name (fallback heuristics)
//    * Sends enemies_n, (optional) enemies_pos[] (flattened x,y,z), nearest_enemy_dist,
//      nearest_enemy_dir (local XZ), plus a "refresh_enemies" command.
// - "camera/cc/rb/player/auto" tracking preference is preserved.
// - Same progress/finish reward as before.

using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Diagnostics;
using HarmonyLib;
using MelonLoader;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.Collections.Generic;

[assembly: MelonInfo(typeof(NeonRLBridge.NeonRLBridge), "NeonRLBridge", "1.1.0", "your-name")]
[assembly: MelonGame(null, null)] // allow all, or set to Neon White's identifiers

namespace NeonRLBridge
{
    [Serializable]
    public class ObsPacket
    {
        public string type = "obs";
        public float[] pos;       // world
        public float[] vel;       // world
        public float yaw_deg;     // player/camera yaw
        public float goal_dist;   // world distance to goal
        public float[] goal_dir;  // local XZ [right, forward] (normalized)
        public bool grounded;
        public int surface;       // 0=air, 1=ground, 2=water
        public float height_gap;  // goal.y - player.y
        public bool reached;
        public int stage;
        public string death_reason;
        public float reward;
        public bool done;

        // Diagnostics/telemetry
        public int kills_total;      // 0 for now
        public float time_unscaled;  // Unity Time.unscaledTime

        // --- NEW: enemy telemetry ---
        public int enemies_n;             // number of live enemies detected
        public float[] enemies_pos;       // flattened [x,y,z]*M (M<=NW_ENEMY_MAX_SEND), optional
        public float nearest_enemy_dist;  // distance to nearest enemy (world), -1 if none
        public float[] nearest_enemy_dir; // local XZ [right, forward] unit dir toward nearest enemy
    }

    [Serializable]
    public class ReadyPacket
    {
        public string type = "ready";
        public string level;
    }

    [Serializable]
    public class CommandPacket
    {
        public string type;
        public string name;
        public string level;
        public float value;
        public int stage;
        public float[] move;
        public float[] look;
        public bool jump;
        public bool shoot;
        public bool use;
        public bool reset;
    }

    [Serializable]
    public class EventPacket
    {
        public string type = \"event\";
        public string name;
        public string payload;
    }

    public class NeonRLBridge : MelonMod
    {
        // ---- Config (env-driven) ----
        int PORT = ParseIntEnv("NW_PORT", 5555);

        float FINISH_RADIUS  = ParseFloatEnv("NW_REACH_RADIUS", 1.5f);
        float PROGRESS_GAIN  = ParseFloatEnv("NW_W_PROGRESS",  8.0f);
        float PROGRESS_CLAMP = ParseFloatEnv("NW_PROG_CLAMP_M", 3.0f);
        float STEP_PENALTY   = ParseFloatEnv("NW_STEP_PENALTY", 0.0f);
        float FINISH_BONUS   = ParseFloatEnv("NW_FINISH_BONUS", 2000.0f);

        string FORCE_LEVEL   = Environment.GetEnvironmentVariable("NW_FORCE_LEVEL");
        bool DEBUG_LOG       = GetBoolEnv("NW_BRIDGE_DEBUG", false);

        // Player/goal find hints
        string HINT_PLAYER   = Environment.GetEnvironmentVariable("NW_PLAYER_NAME"); // optional
        string HINT_GOAL     = Environment.GetEnvironmentVariable("NW_GOAL_NAME");   // optional

        // Optional: force an absolute goal position "x,y,z"
        string FORCE_GOAL_XYZ = Environment.GetEnvironmentVariable("NW_FORCE_GOAL_XYZ");
        bool IGNORE_DONE = GetBoolEnv("NW_IGNORE_DONE", false); // safety switch if needed

        // Water detection
        string WATER_TAG = Environment.GetEnvironmentVariable("NW_WATER_TAG") ?? "Water";
        string WATER_LAYER_NAME = Environment.GetEnvironmentVariable("NW_WATER_LAYER"); // optional

        // Env-configurable send rate
        float TARGET_SEND_HZ = ParseFloatEnv("NW_BRIDGE_SEND_HZ", 60f);
        float ACTION_TIMEOUT_SEC = ParseFloatEnv("NW_ACTION_TIMEOUT_SEC", 0.35f);
        float LOOK_DELTA_CLAMP = ParseFloatEnv("NW_LOOK_CLAMP", 5.0f);

        // Preferred tracking source: camera | cc | rb | player | auto
        string TRACK_PREF = (Environment.GetEnvironmentVariable("NW_TRACK_PREFERRED") ?? "camera").Trim().ToLowerInvariant();

        // ---- NEW: enemy discovery tunables ----
        string ENEMY_TAG = Environment.GetEnvironmentVariable("NW_ENEMY_TAG") ?? "Enemy";
        string ENEMY_LAYER_NAME = Environment.GetEnvironmentVariable("NW_ENEMY_LAYER"); // optional
        string ENEMY_NAME_HINTS = Environment.GetEnvironmentVariable("NW_ENEMY_NAME_HINTS"); // e.g. "Enemy,Angel,Demon"
        int ENEMY_MAX_SEND = ParseIntEnv("NW_ENEMY_MAX_SEND", 16);
        float ENEMY_REFRESH_SEC = ParseFloatEnv("NW_ENEMY_REFRESH_SEC", 3.0f);
        bool ENEMY_AUTO_REFRESH = GetBoolEnv("NW_ENEMY_AUTO_REFRESH", true);

        // ---- Runtime state ----
        TcpListener listener;
        TcpClient client;
        StreamReader reader;
        StreamWriter writer;
        Thread netThread;
        volatile bool running = false;

        Transform player;              // main player transform (root we find)
        CharacterController cc;        // if present
        Rigidbody rb;                  // if present (for velocity)
        Transform playerCamera;        // yaw frame

        Transform goal;                // goal transform

        // the transform whose motion we actually track for pos/vel (camera/CC/etc.)
        Transform tracked;

        // Forced goal position (if provided via env)
        bool hasForcedGoal = false;
        Vector3 forcedGoalPos;

        float lastDist = -1f;          // for progress reward
        Vector3 lastPos;               // for velocity when no rigidbody
        float lastPosTime;

        int stage = 0;                 // user-controlled
        bool reachedSentDone = false;

        // Ground/water sensing
        bool isGrounded = false;
        bool onWater = false;

        // Reward pacing / send pacing
        float sendHz = 60f;
        float sendAccum = 0f;

        // Stuck detection
        Vector3 lastTrackedPos;
        Vector3 lastCamPos;
        float lastStuckCheckT;

        // Thread-safe queue lock for writer
        readonly object writeLock = new object();

        // One-time warnings
        bool warnedNoPlayer = false;
        bool warnedNoGoal = false;

        // 1 Hz position logging
        float lastPosLogT = 0f;
        float posLogInterval = 1f;

        // Optional extra telemetry
        int killsTotal = 0;

        // --- NEW: enemy cache ---
        readonly List<Transform> enemies = new List<Transform>();
        float lastEnemyRefreshT = 0f;

        public override void OnInitializeMelon()
        {
            MelonLogger.Msg($"NeonRLBridge starting on port {PORT}...");
            ParseForcedGoal();

            // clamp and apply send rate from env
            sendHz = Mathf.Clamp(TARGET_SEND_HZ, 5f, 120f);
            MelonLogger.Msg($"[NeonRLBridge] sendHz = {sendHz:F1}, track_pref = {TRACK_PREF}");

            InputPatches.Install();
            InputOverride.Configure(ACTION_TIMEOUT_SEC);
            InputOverride.Clear();

            StartServer();
            SceneManager.sceneLoaded += OnSceneLoaded;
        }

        void ParseForcedGoal()
        {
            try
            {
                if (string.IsNullOrWhiteSpace(FORCE_GOAL_XYZ)) return;
                var parts = FORCE_GOAL_XYZ.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 3
                    && float.TryParse(parts[0], out float gx)
                    && float.TryParse(parts[1], out float gy)
                    && float.TryParse(parts[2], out float gz))
                {
                    forcedGoalPos = new Vector3(gx, gy, gz);
                    hasForcedGoal = true;
                    MelonLogger.Msg($"[NeonRLBridge] Using forced goal at {forcedGoalPos} (NW_FORCE_GOAL_XYZ).");
                }
            }
            catch { }
        }

        public override void OnApplicationQuit()
        {
            running = false;
            try { listener?.Stop(); } catch { }
            try { client?.Close();  } catch { }
            InputPatches.Remove();
            InputOverride.SetConnectionState(false);
            InputOverride.Clear();
        }

        void OnSceneLoaded(Scene s, LoadSceneMode mode)
        {
            // Re-find key objects next frame
            MelonCoroutines.Start(DeferredFindStuff());
        }

        System.Collections.IEnumerator DeferredFindStuff()
        {
            yield return null; // wait a frame
            FindPlayer();
            FindGoal();
            RefreshEnemies();  // NEW
            lastDist = -1f;
            reachedSentDone = false;
            warnedNoPlayer = false;
            warnedNoGoal = false;
            SendReady(); // tell the Python side level is ready
        }

        void StartServer()
        {
            running = true;
            netThread = new Thread(NetLoop) { IsBackground = true };
            netThread.Start();
        }

        void NetLoop()
        {
            while (running)
            {
                try
                {
                    listener = new TcpListener(IPAddress.Loopback, PORT);
                    listener.Start();
                    if (DEBUG_LOG) MelonLogger.Msg("Waiting for RL client...");
                    client = listener.AcceptTcpClient();
                    InputOverride.SetConnectionState(true);
                    InputOverride.Clear();
                    using (var ns = client.GetStream())
                    {
                        reader = new StreamReader(ns, Encoding.UTF8);
                        writer = new StreamWriter(ns, new UTF8Encoding(false)) { AutoFlush = true };

                        // greet
                        SendReady();

                        // read commands loop
                        string line;
                        while (running && client.Connected && (line = reader.ReadLine()) != null)
                        {
                            if (string.IsNullOrWhiteSpace(line)) continue;
                            try
                            {
                                var cmd = JsonUtility.FromJson<CommandPacket>(line);
                                if (cmd != null && cmd.type == "command")
                                    HandleCommand(cmd);
                            }
                            catch (Exception e)
                            {
                                if (DEBUG_LOG) MelonLogger.Warning($"Bad command json: {e}");
                            }
                        }
                    }
                }
                catch (Exception e)
                {
                    if (DEBUG_LOG) MelonLogger.Warning($"NetLoop exception: {e.Message}");
                }
                finally
                {
                    try { client?.Close(); } catch { }
                    try { listener?.Stop(); } catch { }
                    client = null; listener = null; reader = null; writer = null;
                    InputOverride.SetConnectionState(false);
                    InputOverride.Clear();
                    Thread.Sleep(500);
                }
            }
        }

        void HandleCommand(CommandPacket cmd)
        {
            if (DEBUG_LOG) MelonLogger.Msg($"Command: {cmd.name}");
            switch (cmd.name)
            {
                case "timescale":
                    Time.timeScale = Mathf.Clamp(cmd.value, 0.0f, 50.0f);
                    break;
                case "set_stage":
                    stage = cmd.stage;
                    break;
                case "action":
                    HandleActionCommand(cmd);
                    break;
                case "load_level":
                    if (!string.IsNullOrEmpty(cmd.level))
                    {
                        reachedSentDone = false;
                        SceneManager.LoadSceneAsync(cmd.level);
                    }
                    break;
                case "refresh_enemies": // NEW
                    RefreshEnemies();
                    break;
                default:
                    break;
            }
        }

        void HandleActionCommand(CommandPacket cmd)
        {
            var action = new InputOverride.ActionCommandData();

            if (cmd.move != null && cmd.move.Length >= 2)
            {
                float strafe = float.IsNaN(cmd.move[0]) ? 0f : cmd.move[0];
                float forward = float.IsNaN(cmd.move[1]) ? 0f : cmd.move[1];
                action.moveRight = Mathf.Clamp(strafe, -1f, 1f);
                action.moveForward = Mathf.Clamp(forward, -1f, 1f);
            }
            else
            {
                action.moveRight = 0f;
                action.moveForward = 0f;
            }

            if (cmd.look != null && cmd.look.Length >= 2)
            {
                float lookX = float.IsNaN(cmd.look[0]) ? 0f : cmd.look[0];
                float lookY = float.IsNaN(cmd.look[1]) ? 0f : cmd.look[1];
                action.lookX = Mathf.Clamp(lookX, -LOOK_DELTA_CLAMP, LOOK_DELTA_CLAMP);
                action.lookY = Mathf.Clamp(lookY, -LOOK_DELTA_CLAMP, LOOK_DELTA_CLAMP);
            }
            else
            {
                action.lookX = 0f;
                action.lookY = 0f;
            }

            action.jump = cmd.jump;
            action.shoot = cmd.shoot;
            action.use = cmd.use;
            action.reset = cmd.reset;

            InputOverride.ReceiveAction(action);
        }

        public override void OnLateUpdate()
        {
            InputOverride.FrameTick();

            if (Input.GetKeyDown(KeyCode.Escape))
            {
                SendEvent("esc_pressed");
            }

            // Try to sustain ~sendHz messages/sec
            sendAccum += Time.unscaledDeltaTime;
            float step = 1f / Mathf.Max(1f, sendHz);
            while (sendAccum >= step)
            {
                sendAccum -= step;
                SendObs();
            }

            // Periodic enemy refresh if enabled
            if (ENEMY_AUTO_REFRESH && (Time.unscaledTime - lastEnemyRefreshT) >= ENEMY_REFRESH_SEC)
            {
                lastEnemyRefreshT = Time.unscaledTime;
                RefreshEnemies();
            }

            // 1 Hz coordinate log (use tracked)
            if (tracked != null && (Time.unscaledTime - lastPosLogT) >= posLogInterval)
            {
                lastPosLogT = Time.unscaledTime;
                var p = tracked.position;
                MelonLogger.Msg($"[Pos] {p.x:F2}, {p.y:F2}, {p.z:F2}");
            }
        }

        void SendReady()
        {
            if (writer == null) return;
            var ready = new ReadyPacket { level = SceneManager.GetActiveScene().name };
            string js = JsonUtility.ToJson(ready);
            SafeWrite(js);
        }

        void SendEvent(string eventName, string payload = null)
        {
            if (writer == null) return;
            var evt = new EventPacket { name = eventName, payload = payload };
            SafeWrite(JsonUtility.ToJson(evt));
        }

        void SendObs()
        {
            if (player == null)
            {
                // keep trying every send tick until found
                FindPlayer();
                if (player == null)
                {
                    // still nothing â†’ send a minimal heartbeat so Python stays alive
                    var pktHeartbeat = new ObsPacket {
                        pos = new float[] {0f,0f,0f},
                        vel = new float[] {0f,0f,0f},
                        yaw_deg = 0f,
                        goal_dist = 9999f,
                        goal_dir = new float[] {0f,1f},
                        grounded = false,
                        surface = 0,
                        height_gap = 0f,
                        reached = false,
                        stage = stage,
                        death_reason = "no_player",
                        reward = 0f,
                        done = false,
                        kills_total = killsTotal,
                        time_unscaled = Time.unscaledTime,

                        enemies_n = 0,
                        enemies_pos = null,
                        nearest_enemy_dist = -1f,
                        nearest_enemy_dir = new float[] {0f,1f}
                    };
                    SafeWrite(JsonUtility.ToJson(pktHeartbeat));
                    return;
                }
            }

            if (writer == null) return;

            if (player == null) FindPlayer();
            EnsureTrackedValid();

            if (goal == null && !hasForcedGoal) FindGoal();

            // Position source: tracked -> camera -> player -> zero
            Transform tpos = tracked != null ? tracked : (playerCamera != null ? playerCamera : player);
            Vector3 p = tpos != null ? tpos.position : Vector3.zero;
            Vector3 v = EstimateVelocity();
            float yaw = GetYawDegrees();

            // Determine goal position with safe fallback
            bool usingFallback = false;
            Vector3 gpos;
            if (hasForcedGoal)
            {
                gpos = forcedGoalPos;
            }
            else if (goal != null)
            {
                gpos = goal.position;
            }
            else
            {
                // Fallback: "fake" goal far ahead so dist>0 and we never reach
                usingFallback = true;
                if (!warnedNoGoal)
                {
                    MelonLogger.Warning("NeonRLBridge: Goal not found. Using far-ahead fallback. Set NW_GOAL_NAME or NW_FORCE_GOAL_XYZ to fix.");
                    warnedNoGoal = true;
                }
                Vector3 fwd = (playerCamera != null ? playerCamera.forward :
                               tracked != null ? tracked.forward :
                               player != null ? player.forward : Vector3.forward);
                gpos = p + fwd.normalized * 1000f;
            }

            float dist = Vector3.Distance(p, gpos);
            Vector2 goalDirLocal = ComputeLocalDir(p, yaw, gpos);
            float heightGap = (gpos.y - p.y);

            // contact sensing
            isGrounded = SenseGrounded();
            onWater = SenseWater();

            // REWARD: potential progress + finish bonus + (optional step penalty)
            float progress = 0f;
            if (lastDist >= 0f && !usingFallback)
            {
                float rawDelta = lastDist - dist;
                float delta = Mathf.Clamp(Mathf.Max(0f, rawDelta), 0f, PROGRESS_CLAMP);
                progress = delta * PROGRESS_GAIN;
            }
            lastDist = dist; // keep updated

            bool reached = (!usingFallback) && (dist <= FINISH_RADIUS);
            bool done = false;
            float reward = progress - STEP_PENALTY;

            if (reached && !reachedSentDone)
            {
                reward += FINISH_BONUS;
                done = true;         // tell the env we reached goal
                reachedSentDone = true;
            }

            if (IGNORE_DONE) done = false; // hard override if needed

            int surface = 0;
            if (onWater) surface = 2;
            else if (isGrounded) surface = 1;

            // ----- NEW: enemy telemetry -----
            int enemiesN;
            float nearestDist;
            Vector2 nearestLocalDir;
            float[] enemyPosOut;
            CollectEnemyTelemetry(p, yaw, out enemiesN, out nearestDist, out nearestLocalDir, out enemyPosOut);

            var pkt = new ObsPacket
            {
                pos = new float[] { p.x, p.y, p.z },
                vel = new float[] { v.x, v.y, v.z },
                yaw_deg = yaw,
                goal_dist = dist,
                goal_dir = new float[] { goalDirLocal.x, goalDirLocal.y },
                grounded = isGrounded,
                surface = surface,
                height_gap = heightGap,
                reached = reached,
                stage = stage,
                death_reason = null,
                reward = reward,
                done = done,

                // telemetry
                kills_total = killsTotal,
                time_unscaled = Time.unscaledTime,

                enemies_n = enemiesN,
                enemies_pos = enemyPosOut,
                nearest_enemy_dist = nearestDist,
                nearest_enemy_dir = new float[] { nearestLocalDir.x, nearestLocalDir.y }
            };

            string js = JsonUtility.ToJson(pkt);
            SafeWrite(js);
        }

        // ---- helpers ----

        void SafeWrite(string s)
        {
            try
            {
                lock (writeLock)
                {
                    writer.WriteLine(s);
                }
            }
            catch { /* ignore */ }
        }

        // Choose/validate the transform that actually moves
        Transform ResolveTracked()
        {
            // Explicit preference
            switch (TRACK_PREF)
            {
                case "camera":
                    if (playerCamera != null) return playerCamera;
                    break;
                case "cc":
                    if (cc != null) return cc.transform;
                    break;
                case "rb":
                    if (rb != null) return rb.transform;
                    break;
                case "player":
                    if (player != null) return player;
                    break;
                case "auto":
                default:
                    if (playerCamera != null) return playerCamera;
                    if (cc != null) return cc.transform;
                    if (rb != null) return rb.transform;
                    if (player != null) return player;
                    break;
            }
            return null;
        }

        void EnsureTrackedValid()
        {
            // (Re)resolve if null
            if (tracked == null) tracked = ResolveTracked();

            // One-time init of baselines
            if (lastPosTime <= 0f) lastPosTime = Time.time;
            if (tracked != null && lastPos == default(Vector3)) lastPos = tracked.position;
            if (playerCamera != null && lastCamPos == default(Vector3)) lastCamPos = playerCamera.position;
            if (tracked != null && lastTrackedPos == default(Vector3)) lastTrackedPos = tracked.position;

            // Periodically check for "stuck" tracked while camera moves; switch to camera
            float now = Time.time;
            if (now - lastStuckCheckT > 0.25f) // 4Hz check is enough
            {
                lastStuckCheckT = now;
                Vector3 curTracked = tracked != null ? tracked.position : Vector3.zero;
                Vector3 curCam = playerCamera != null ? playerCamera.position : curTracked;

                float trackedMove = (curTracked - lastTrackedPos).sqrMagnitude;
                float camMove = (curCam - lastCamPos).sqrMagnitude;

                lastTrackedPos = curTracked;
                lastCamPos = curCam;

                // If tracked isn't moving but camera is, and we have a camera, switch
                if (playerCamera != null && trackedMove < 1e-8f && camMove > 1e-8f && tracked != playerCamera)
                {
                    tracked = playerCamera;
                    if (DEBUG_LOG)
                        MelonLogger.Msg("[NeonRLBridge] Tracked transform appeared static; switching to playerCamera.");
                }
            }
        }

        void FindPlayer()
        {
            player = null; cc = null; rb = null; tracked = null;

            // 0) quick explicit name from env
            if (!string.IsNullOrEmpty(HINT_PLAYER))
            {
                var go = GameObject.Find(HINT_PLAYER);
                if (go != null) player = go.transform;
            }

            // 0.5) common Neon White name seen in logs
            if (player == null)
            {
                var go = GameObject.Find("PlayerMouse_Player0");
                if (go != null) player = go.transform;
            }

            // 1) try tag "Player"
            if (player == null)
            {
                try
                {
                    var tagged = GameObject.FindGameObjectsWithTag("Player");
                    if (tagged != null && tagged.Length > 0)
                        player = tagged[0].transform;
                }
                catch { }
            }

            // 2) nearest CharacterController to main camera
            var cam = Camera.main;
            if (cam != null) playerCamera = cam.transform;

            if (player == null && cam != null)
            {
                var ccs = GameObject.FindObjectsOfType<CharacterController>();
                float best = float.MaxValue;
                foreach (var c in ccs)
                {
                    float d = Vector3.Distance(cam.transform.position, c.transform.position);
                    if (d < best) { best = d; player = c.transform; cc = c; }
                }
            }

            // 3) any CharacterController
            if (player == null)
            {
                var ccs = GameObject.FindObjectsOfType<CharacterController>();
                if (ccs.Length > 0) { cc = ccs[0]; player = cc.transform; }
            }

            // 4) any Rigidbody that has a Camera child
            if (player == null)
            {
                var rbs = GameObject.FindObjectsOfType<Rigidbody>();
                foreach (var r in rbs)
                {
                    if (r.GetComponentInChildren<Camera>() != null)
                    { rb = r; player = r.transform; break; }
                }
            }

            // 5) final fallback: Camera.main
            if (player == null && Camera.main != null)
            {
                player = Camera.main.transform;
                playerCamera = Camera.main.transform;
                MelonLogger.Msg("[NeonRLBridge] Using Camera.main as player fallback.");
            }

            if (player == null)
            {
                if (!warnedNoPlayer)
                {
                    MelonLogger.Warning("NeonRLBridge: Player not found (set NW_PLAYER_NAME to help).");
                    warnedNoPlayer = true;
                    DumpQuickCandidates();
                }
                return;
            }

            tracked = ResolveTracked();

            lastPosTime = Time.time;
            if (tracked != null) lastPos = tracked.position; else lastPos = player.position;
            if (playerCamera == null && Camera.main != null) playerCamera = Camera.main.transform;

            string trackedName = (tracked != null ? tracked.name : "null");
            Vector3 trackedPos = (tracked != null ? tracked.position : Vector3.zero);
            MelonLogger.Msg($"[NeonRLBridge] Using player '{player.name}' @ {player.position} (tracking '{trackedName}' @ {trackedPos})");
        }

        void DumpQuickCandidates()
        {
            var cams = GameObject.FindObjectsOfType<Camera>();
            MelonLogger.Msg($"[Dump] Cameras: {cams.Length}");
            foreach (var c in cams)
                MelonLogger.Msg($"  Cam: {c.name} (Main={(c == Camera.main)}) @ {c.transform.position}");

            var t = GameObject.Find("PlayerMouse_Player0");
            if (t != null) MelonLogger.Msg("  Candidate: PlayerMouse_Player0 @ " + t.transform.position);
        }

        void FindGoal()
        {
            if (hasForcedGoal) { goal = null; return; } // forced xyz wins
            goal = null;

            // 1) by hint
            if (!string.IsNullOrEmpty(HINT_GOAL))
            {
                var go = GameObject.Find(HINT_GOAL);
                if (go != null) goal = go.transform;
            }

            // 2) tags "Finish" or "Goal"
            if (goal == null)
            {
                try
                {
                    var finish = GameObject.FindGameObjectsWithTag("Finish");
                    if (finish != null && finish.Length > 0) goal = finish[0].transform;
                }
                catch { }
                if (goal == null)
                {
                    try
                    {
                        var goalTagged = GameObject.FindGameObjectsWithTag("Goal");
                        if (goalTagged != null && goalTagged.Length > 0) goal = goalTagged[0].transform;
                    }
                    catch { }
                }
            }

            // 3) name search
            if (goal == null)
            {
                var all = GameObject.FindObjectsOfType<Transform>();
                float best = float.MaxValue;
                foreach (var t in all)
                {
                    string n = (t.name ?? "").ToLower();
                    if (n.Contains("goal") || n.Contains("finish") || n.Contains("exit") || n.Contains("end"))
                    {
                        float d = (player != null) ? Vector3.Distance(player.position, t.position) : 0f;
                        if (d < best) { best = d; goal = t; }
                    }
                }
            }
        }

        // --- NEW: enemy discovery & telemetry ---
        void RefreshEnemies()
        {
            enemies.Clear();

            var nameHints = new List<string>();
            if (!string.IsNullOrEmpty(ENEMY_NAME_HINTS))
            {
                foreach (var token in ENEMY_NAME_HINTS.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
                    nameHints.Add(token.Trim().ToLower());
            }

            int enemyLayer = -1;
            if (!string.IsNullOrEmpty(ENEMY_LAYER_NAME))
                enemyLayer = LayerMask.NameToLayer(ENEMY_LAYER_NAME);

            Transform[] all = GameObject.FindObjectsOfType<Transform>();
            var uniq = new HashSet<GameObject>();

            foreach (var t in all)
            {
                if (t == null) continue;
                var go = t.gameObject;
                if (go == null) continue;
                if (!go.activeInHierarchy) continue;

                bool match = false;

                // Tag
                if (!string.IsNullOrEmpty(ENEMY_TAG))
                {
                    try { if (go.CompareTag(ENEMY_TAG)) match = true; }
                    catch { /* tag may not exist in this scene */ }
                }

                // Layer
                if (!match && enemyLayer >= 0 && go.layer == enemyLayer) match = true;

                // Name hints
                if (!match && nameHints.Count > 0)
                {
                    string n = (go.name ?? "").ToLower();
                    foreach (var h in nameHints) { if (n.Contains(h)) { match = true; break; } }
                }

                // Component type name containing "Enemy"
                if (!match)
                {
                    var comps = go.GetComponents<Component>();
                    foreach (var c in comps)
                    {
                        if (c == null) continue;
                        string tn = c.GetType().Name.ToLower();
                        if (tn.Contains("enemy")) { match = true; break; }
                    }
                }

                if (match)
                {
                    // avoid duplicates from child objects by using the topmost with a Rigidbody/Animator/Collider if possible
                    GameObject root = go;
                    // a simple dedupe: keep unique GameObjects
                    if (uniq.Add(root))
                        enemies.Add(t);
                }
            }

            if (DEBUG_LOG)
                MelonLogger.Msg($"[Enemies] Refreshed: {enemies.Count} candidates.");
        }

        void CollectEnemyTelemetry(Vector3 playerPos, float yawDeg,
                                   out int enemiesN,
                                   out float nearestDist,
                                   out Vector2 nearestLocalDir,
                                   out float[] enemiesPosOut)
        {
            enemiesN = 0;
            nearestDist = -1f;
            nearestLocalDir = new Vector2(0f, 1f);
            enemiesPosOut = null;

            if (enemies.Count == 0) return;

            // compact positions for up to ENEMY_MAX_SEND enemies
            int maxSend = Mathf.Max(0, ENEMY_MAX_SEND);
            if (maxSend > 0)
                enemiesPosOut = new float[Mathf.Min(maxSend, enemies.Count) * 3];

            float best = float.MaxValue;
            int write = 0;
            for (int i = enemies.Count - 1; i >= 0; --i)
            {
                var t = enemies[i];
                if (t == null || t.gameObject == null || !t.gameObject.activeInHierarchy)
                {
                    enemies.RemoveAt(i);
                    continue;
                }
            }

            enemiesN = enemies.Count;

            for (int i = 0; i < enemies.Count; ++i)
            {
                var t = enemies[i];
                Vector3 ep = t.position;
                float d = Vector3.Distance(playerPos, ep);
                if (d < best)
                {
                    best = d;
                    nearestDist = d;
                    nearestLocalDir = ComputeLocalDir(playerPos, yawDeg, ep);
                }

                if (enemiesPosOut != null && write + 3 <= enemiesPosOut.Length)
                {
                    enemiesPosOut[write++] = ep.x;
                    enemiesPosOut[write++] = ep.y;
                    enemiesPosOut[write++] = ep.z;
                }
            }
        }

        Vector3 EstimateVelocity()
        {
            if (rb != null) return rb.velocity;

            float t = Time.time;
            float dt = Mathf.Max(1e-4f, t - lastPosTime);

            Transform tpos = tracked != null ? tracked : (playerCamera != null ? playerCamera : player);
            Vector3 cur = tpos != null ? tpos.position : Vector3.zero;

            Vector3 v = (cur - lastPos) / dt;
            lastPos = cur;
            lastPosTime = t;
            return v;
        }

        float GetYawDegrees()
        {
            // Prefer camera yaw (often reflects look dir), else tracked, else player
            Transform t = playerCamera != null ? playerCamera : (tracked != null ? tracked : player);
            if (t == null) return 0f;
            return t.eulerAngles.y;
        }

        Vector2 ComputeLocalDir(Vector3 playerPos, float yawDeg, Vector3 worldTarget)
        {
            Vector3 to = worldTarget - playerPos;
            Vector2 xz = new Vector2(to.x, to.z);
            if (xz.sqrMagnitude < 1e-8f) return Vector2.zero;

            float th = yawDeg * Mathf.Deg2Rad;
            float c = Mathf.Cos(th), s = Mathf.Sin(th);
            // rotate world into local (right, forward)
            float lx =  c * xz.x + s * xz.y;
            float lz = -s * xz.x + c * xz.y;
            Vector2 local = new Vector2(lx, lz);
            return local.normalized;
        }

        bool SenseGrounded()
        {
            if (cc != null) return cc.isGrounded;

            Transform tpos = tracked != null ? tracked : (player != null ? player : null);
            if (tpos == null) return false;

            // Raycast down from tracked (or player)
            Vector3 origin = tpos.position + Vector3.up * 0.1f;
            if (Physics.Raycast(origin, Vector3.down, out RaycastHit hit, 0.3f, ~0, QueryTriggerInteraction.Ignore))
                return true;
            return false;
        }

        bool SenseWater()
        {
            Transform tpos = tracked != null ? tracked : (player != null ? player : null);
            if (tpos == null) return false;

            // Check overlap at feet for colliders tagged/layered as water
            Vector3 center = tpos.position + Vector3.up * 0.05f;
            Collider[] cols = Physics.OverlapSphere(center, 0.25f, ~0, QueryTriggerInteraction.Collide);

            int waterLayer = -1;
            if (!string.IsNullOrEmpty(WATER_LAYER_NAME))
            {
                waterLayer = LayerMask.NameToLayer(WATER_LAYER_NAME);
            }

            foreach (var c in cols)
            {
                if (c == null) continue;
                if (!string.IsNullOrEmpty(WATER_TAG) && c.CompareTag(WATER_TAG)) return true;
                if (waterLayer >= 0 && c.gameObject.layer == waterLayer) return true;

                // Material name heuristic
                var mat = c.sharedMaterial;
                if (mat != null && mat.name != null && mat.name.ToLower().Contains("water"))
                    return true;

                // Name heuristic
                string n = c.gameObject.name.ToLower();
                if (n.Contains("water")) return true;
            }
            return false;
        }

        // ---- utils ----
        static int ParseIntEnv(string k, int defVal)
        {
            string v = Environment.GetEnvironmentVariable(k);
            if (string.IsNullOrEmpty(v)) return defVal;
            if (int.TryParse(v, out int o)) return o;
            return defVal;
        }

        static float ParseFloatEnv(string k, float defVal)
        {
            string v = Environment.GetEnvironmentVariable(k);
            if (string.IsNullOrEmpty(v)) return defVal;
            if (float.TryParse(v, out float o)) return o;
            return defVal;
        }

        static bool GetBoolEnv(string k, bool defVal)
        {
            string v = Environment.GetEnvironmentVariable(k);
            if (string.IsNullOrEmpty(v)) return defVal;
            v = v.Trim().ToLowerInvariant();
            return !(v == "0" || v == "false" || v == "no" || v == "off" || v == "");
        }
    }
    internal static class InputOverride
    {
        internal enum KeyQuery { Hold, Down, Up }

        internal struct ActionCommandData
        {
            public float moveRight;
            public float moveForward;
            public float lookX;
            public float lookY;
            public bool jump;
            public bool shoot;
            public bool use;
            public bool reset;
        }

        struct FrameSnapshot
        {
            public bool active;
            public float moveRight;
            public float moveForward;
            public float lookX;
            public float lookY;
            public bool jump;
            public bool jumpDown;
            public bool jumpUp;
            public bool shoot;
            public bool shootDown;
            public bool shootUp;
            public bool use;
            public bool useDown;
            public bool useUp;
            public bool reset;
            public bool resetDown;
            public bool resetUp;
            public bool w;
            public bool wDown;
            public bool wUp;
            public bool a;
            public bool aDown;
            public bool aUp;
            public bool s;
            public bool sDown;
            public bool sUp;
            public bool d;
            public bool dDown;
            public bool dUp;
        }

        enum ButtonKind { Unknown, Jump, Shoot, Use, Reset }
        enum MovementKey { None, Forward, Backward, Left, Right }
        enum AxisKind { None, MoveRight, MoveForward, LookX, LookY }

        static readonly object Sync = new object();
        static ActionCommandData desired;
        static FrameSnapshot current;
        static bool connectionActive;
        static bool overrideActive;
        static long lastActionTick;
        static double overrideTimeout = 0.35f;
        static int lastFrameProcessed = -1;
        const float MovementThreshold = 0.35f;
        static readonly double TickFrequency = Stopwatch.Frequency;

        public static void Configure(float timeoutSec)
        {
            lock (Sync)
            {
                if (timeoutSec <= 0.01f)
                {
                    timeoutSec = 0.01f;
                }
                overrideTimeout = timeoutSec;
            }
        }

        public static void SetConnectionState(bool connected)
        {
            lock (Sync)
            {
                connectionActive = connected;
                if (!connected)
                {
                    overrideActive = false;
                    desired = default;
                    current = default;
                    lastActionTick = 0;
                    lastFrameProcessed = -1;
                }
            }
        }

        public static void Clear()
        {
            lock (Sync)
            {
                desired = default;
                current = default;
                overrideActive = false;
                lastActionTick = 0;
                lastFrameProcessed = -1;
            }
        }

        public static void ReceiveAction(ActionCommandData data)
        {
            lock (Sync)
            {
                desired.moveRight = Sanitize(data.moveRight, -1f, 1f);
                desired.moveForward = Sanitize(data.moveForward, -1f, 1f);
                desired.lookX = Sanitize(data.lookX, -float.MaxValue, float.MaxValue);
                desired.lookY = Sanitize(data.lookY, -float.MaxValue, float.MaxValue);
                desired.jump = data.jump;
                desired.shoot = data.shoot;
                desired.use = data.use;
                desired.reset = data.reset;
                lastActionTick = Stopwatch.GetTimestamp();
                overrideActive = true;
            }
        }

        public static void FrameTick()
        {
            lock (Sync)
            {
                EnsureFrameLocked();
            }
        }

        public static bool TryGetAxis(string axisName, out float value)
        {
            value = 0f;
            if (string.IsNullOrWhiteSpace(axisName))
            {
                return false;
            }

            string key = axisName.Trim().ToLowerInvariant();

            lock (Sync)
            {
                EnsureFrameLocked();
                if (!current.active)
                {
                    return false;
                }

                AxisKind kind = key switch
                {
                    "horizontal" => AxisKind.MoveRight,
                    "move_horizontal" => AxisKind.MoveRight,
                    "movehorizontal" => AxisKind.MoveRight,
                    "strafe" => AxisKind.MoveRight,
                    "strafeaxis" => AxisKind.MoveRight,
                    "strafe_x" => AxisKind.MoveRight,
                    "vertical" => AxisKind.MoveForward,
                    "move_vertical" => AxisKind.MoveForward,
                    "movevertical" => AxisKind.MoveForward,
                    "forward" => AxisKind.MoveForward,
                    "moveforward" => AxisKind.MoveForward,
                    "mouse x" => AxisKind.LookX,
                    "mousex" => AxisKind.LookX,
                    "look x" => AxisKind.LookX,
                    "lookx" => AxisKind.LookX,
                    "cam x" => AxisKind.LookX,
                    "camera x" => AxisKind.LookX,
                    "mouse y" => AxisKind.LookY,
                    "mousey" => AxisKind.LookY,
                    "look y" => AxisKind.LookY,
                    "looky" => AxisKind.LookY,
                    "cam y" => AxisKind.LookY,
                    "camera y" => AxisKind.LookY,
                    _ => AxisKind.None
                };

                switch (kind)
                {
                    case AxisKind.MoveRight:
                        value = current.moveRight;
                        return true;
                    case AxisKind.MoveForward:
                        value = current.moveForward;
                        return true;
                    case AxisKind.LookX:
                        value = current.lookX;
                        return true;
                    case AxisKind.LookY:
                        value = current.lookY;
                        return true;
                    default:
                        return false;
                }
            }
        }

        public static bool TryGetKey(KeyCode keyCode, KeyQuery query, out bool value)
        {
            lock (Sync)
            {
                EnsureFrameLocked();
                if (!current.active)
                {
                    value = false;
                    return false;
                }

                switch (keyCode)
                {
                    case KeyCode.W:
                        value = QueryMovement(query, MovementKey.Forward, current);
                        return true;
                    case KeyCode.S:
                        value = QueryMovement(query, MovementKey.Backward, current);
                        return true;
                    case KeyCode.A:
                        value = QueryMovement(query, MovementKey.Left, current);
                        return true;
                    case KeyCode.D:
                        value = QueryMovement(query, MovementKey.Right, current);
                        return true;
                    case KeyCode.Space:
                        value = QueryButton(ButtonKind.Jump, query, current);
                        return true;
                    case KeyCode.Mouse0:
                        value = QueryButton(ButtonKind.Shoot, query, current);
                        return true;
                    case KeyCode.Mouse1:
                        value = QueryButton(ButtonKind.Use, query, current);
                        return true;
                    case KeyCode.F:
                        value = QueryButton(ButtonKind.Reset, query, current);
                        return true;
                    default:
                        value = false;
                        return false;
                }
            }
        }

        public static bool TryGetKey(string keyName, KeyQuery query, out bool value)
        {
            value = false;
            if (string.IsNullOrWhiteSpace(keyName))
            {
                return false;
            }

            if (!TryParseKeyName(keyName, out var keyCode))
            {
                return false;
            }

            return TryGetKey(keyCode, query, out value);
        }

        public static bool TryGetButton(string buttonName, KeyQuery query, out bool value)
        {
            value = false;
            if (string.IsNullOrWhiteSpace(buttonName))
            {
                return false;
            }

            string key = buttonName.Trim().ToLowerInvariant();
            ButtonKind kind = key switch
            {
                "jump" => ButtonKind.Jump,
                "fire" => ButtonKind.Shoot,
                "fire1" => ButtonKind.Shoot,
                "shoot" => ButtonKind.Shoot,
                "attack" => ButtonKind.Shoot,
                "fire2" => ButtonKind.Use,
                "altfire" => ButtonKind.Use,
                "use" => ButtonKind.Use,
                "interact" => ButtonKind.Use,
                "reset" => ButtonKind.Reset,
                "resetlevel" => ButtonKind.Reset,
                "restart" => ButtonKind.Reset,
                _ => ButtonKind.Unknown
            };

            if (kind == ButtonKind.Unknown)
            {
                return false;
            }

            lock (Sync)
            {
                EnsureFrameLocked();
                if (!current.active)
                {
                    return false;
                }

                value = QueryButton(kind, query, current);
                return true;
            }
        }

        public static bool TryGetMouseButton(int button, KeyQuery query, out bool value)
        {
            ButtonKind kind = button switch
            {
                0 => ButtonKind.Shoot,
                1 => ButtonKind.Use,
                _ => ButtonKind.Unknown
            };

            if (kind == ButtonKind.Unknown)
            {
                value = false;
                return false;
            }

            lock (Sync)
            {
                EnsureFrameLocked();
                if (!current.active)
                {
                    value = false;
                    return false;
                }

                value = QueryButton(kind, query, current);
                return true;
            }
        }

        static void EnsureFrameLocked()
        {
            int frame = Time.frameCount;
            if (frame == lastFrameProcessed)
            {
                return;
            }

            FrameSnapshot previous = current;
            FrameSnapshot next = default;

            double ageSec = double.MaxValue;
            if (lastActionTick != 0)
            {
                ageSec = (Stopwatch.GetTimestamp() - lastActionTick) / TickFrequency;
            }

            bool stale = !connectionActive || !overrideActive || ageSec > overrideTimeout;

            if (!stale)
            {
                next.active = true;
                next.moveRight = desired.moveRight;
                next.moveForward = desired.moveForward;
                next.lookX = desired.lookX;
                next.lookY = desired.lookY;
                next.jump = desired.jump;
                next.shoot = desired.shoot;
                next.use = desired.use;
                next.reset = desired.reset;

                next.w = next.moveForward > MovementThreshold;
                next.s = next.moveForward < -MovementThreshold;
                next.d = next.moveRight > MovementThreshold;
                next.a = next.moveRight < -MovementThreshold;
            }

            next.jumpDown = next.jump && !previous.jump;
            next.jumpUp = !next.jump && previous.jump;
            next.shootDown = next.shoot && !previous.shoot;
            next.shootUp = !next.shoot && previous.shoot;
            next.useDown = next.use && !previous.use;
            next.useUp = !next.use && previous.use;
            next.resetDown = next.reset && !previous.reset;
            next.resetUp = !next.reset && previous.reset;

            next.wDown = next.w && !previous.w;
            next.wUp = !next.w && previous.w;
            next.aDown = next.a && !previous.a;
            next.aUp = !next.a && previous.a;
            next.sDown = next.s && !previous.s;
            next.sUp = !next.s && previous.s;
            next.dDown = next.d && !previous.d;
            next.dUp = !next.d && previous.d;

            current = next;
            lastFrameProcessed = frame;

            desired.lookX = 0f;
            desired.lookY = 0f;

            if (stale)
            {
                overrideActive = false;
                desired.moveRight = 0f;
                desired.moveForward = 0f;
                desired.jump = false;
                desired.shoot = false;
                desired.use = false;
                desired.reset = false;
            }
        }

        static bool QueryMovement(KeyQuery query, MovementKey key, FrameSnapshot frame)
        {
            return query switch
            {
                KeyQuery.Hold => MovementHold(key, frame),
                KeyQuery.Down => MovementDown(key, frame),
                KeyQuery.Up => MovementUp(key, frame),
                _ => false
            };
        }

        static bool MovementHold(MovementKey key, FrameSnapshot frame)
        {
            return key switch
            {
                MovementKey.Forward => frame.w,
                MovementKey.Backward => frame.s,
                MovementKey.Left => frame.a,
                MovementKey.Right => frame.d,
                _ => false
            };
        }

        static bool MovementDown(MovementKey key, FrameSnapshot frame)
        {
            return key switch
            {
                MovementKey.Forward => frame.wDown,
                MovementKey.Backward => frame.sDown,
                MovementKey.Left => frame.aDown,
                MovementKey.Right => frame.dDown,
                _ => false
            };
        }

        static bool MovementUp(MovementKey key, FrameSnapshot frame)
        {
            return key switch
            {
                MovementKey.Forward => frame.wUp,
                MovementKey.Backward => frame.sUp,
                MovementKey.Left => frame.aUp,
                MovementKey.Right => frame.dUp,
                _ => false
            };
        }

        static bool QueryButton(ButtonKind kind, KeyQuery query, FrameSnapshot frame)
        {
            return kind switch
            {
                ButtonKind.Jump => query switch
                {
                    KeyQuery.Hold => frame.jump,
                    KeyQuery.Down => frame.jumpDown,
                    KeyQuery.Up => frame.jumpUp,
                    _ => false
                },
                ButtonKind.Shoot => query switch
                {
                    KeyQuery.Hold => frame.shoot,
                    KeyQuery.Down => frame.shootDown,
                    KeyQuery.Up => frame.shootUp,
                    _ => false
                },
                ButtonKind.Use => query switch
                {
                    KeyQuery.Hold => frame.use,
                    KeyQuery.Down => frame.useDown,
                    KeyQuery.Up => frame.useUp,
                    _ => false
                },
                ButtonKind.Reset => query switch
                {
                    KeyQuery.Hold => frame.reset,
                    KeyQuery.Down => frame.resetDown,
                    KeyQuery.Up => frame.resetUp,
                    _ => false
                },
                _ => false
            };
        }

        static bool TryParseKeyName(string keyName, out KeyCode keyCode)
        {
            string key = keyName.Trim().ToLowerInvariant();
            keyCode = key switch
            {
                "w" => KeyCode.W,
                "a" => KeyCode.A,
                "s" => KeyCode.S,
                "d" => KeyCode.D,
                "space" => KeyCode.Space,
                "spacebar" => KeyCode.Space,
                "jump" => KeyCode.Space,
                "mouse0" => KeyCode.Mouse0,
                "leftmouse" => KeyCode.Mouse0,
                "lmb" => KeyCode.Mouse0,
                "mouse1" => KeyCode.Mouse1,
                "rightmouse" => KeyCode.Mouse1,
                "rmb" => KeyCode.Mouse1,
                "f" => KeyCode.F,
                "reset" => KeyCode.F,
                _ => KeyCode.None
            };
            return keyCode != KeyCode.None;
        }

        static float Sanitize(float value, float min, float max)
        {
            if (float.IsNaN(value) || float.IsInfinity(value))
            {
                return 0f;
            }

            if (min == -float.MaxValue && max == float.MaxValue)
            {
                return value;
            }

            return Mathf.Clamp(value, min, max);
        }
    }

    internal static class InputPatches
    {
        static Harmony harmony;

        public static void Install()
        {
            if (harmony != null)
            {
                return;
            }

            harmony = new Harmony("NeonRLBridge.InputOverride");
            harmony.PatchAll(typeof(InputPatches));
        }

        public static void Remove()
        {
            if (harmony == null)
            {
                return;
            }

            harmony.UnpatchSelf();
            harmony = null;
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetAxisRaw))]
        static class GetAxisRawPatch
        {
            static bool Prefix(string axisName, ref float __result)
            {
                if (InputOverride.TryGetAxis(axisName, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetAxis))]
        static class GetAxisPatch
        {
            static bool Prefix(string axisName, ref float __result)
            {
                if (InputOverride.TryGetAxis(axisName, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetKey), new[] { typeof(KeyCode) })]
        static class GetKeyPatch
        {
            static bool Prefix(KeyCode key, ref bool __result)
            {
                if (InputOverride.TryGetKey(key, InputOverride.KeyQuery.Hold, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetKeyDown), new[] { typeof(KeyCode) })]
        static class GetKeyDownPatch
        {
            static bool Prefix(KeyCode key, ref bool __result)
            {
                if (InputOverride.TryGetKey(key, InputOverride.KeyQuery.Down, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetKeyUp), new[] { typeof(KeyCode) })]
        static class GetKeyUpPatch
        {
            static bool Prefix(KeyCode key, ref bool __result)
            {
                if (InputOverride.TryGetKey(key, InputOverride.KeyQuery.Up, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetKey), new[] { typeof(string) })]
        static class GetKeyStringPatch
        {
            static bool Prefix(string name, ref bool __result)
            {
                if (InputOverride.TryGetKey(name, InputOverride.KeyQuery.Hold, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetKeyDown), new[] { typeof(string) })]
        static class GetKeyDownStringPatch
        {
            static bool Prefix(string name, ref bool __result)
            {
                if (InputOverride.TryGetKey(name, InputOverride.KeyQuery.Down, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetKeyUp), new[] { typeof(string) })]
        static class GetKeyUpStringPatch
        {
            static bool Prefix(string name, ref bool __result)
            {
                if (InputOverride.TryGetKey(name, InputOverride.KeyQuery.Up, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetButton))]
        static class GetButtonPatch
        {
            static bool Prefix(string buttonName, ref bool __result)
            {
                if (InputOverride.TryGetButton(buttonName, InputOverride.KeyQuery.Hold, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetButtonDown))]
        static class GetButtonDownPatch
        {
            static bool Prefix(string buttonName, ref bool __result)
            {
                if (InputOverride.TryGetButton(buttonName, InputOverride.KeyQuery.Down, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetButtonUp))]
        static class GetButtonUpPatch
        {
            static bool Prefix(string buttonName, ref bool __result)
            {
                if (InputOverride.TryGetButton(buttonName, InputOverride.KeyQuery.Up, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetMouseButton))]
        static class GetMouseButtonPatch
        {
            static bool Prefix(int button, ref bool __result)
            {
                if (InputOverride.TryGetMouseButton(button, InputOverride.KeyQuery.Hold, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetMouseButtonDown))]
        static class GetMouseButtonDownPatch
        {
            static bool Prefix(int button, ref bool __result)
            {
                if (InputOverride.TryGetMouseButton(button, InputOverride.KeyQuery.Down, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }

        [HarmonyPatch(typeof(Input), nameof(Input.GetMouseButtonUp))]
        static class GetMouseButtonUpPatch
        {
            static bool Prefix(int button, ref bool __result)
            {
                if (InputOverride.TryGetMouseButton(button, InputOverride.KeyQuery.Up, out var value))
                {
                    __result = value;
                    return false;
                }

                return true;
            }
        }
    }
}











