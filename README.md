# DEEPRL — Live PPO: Soccer Arena + Drone Course

> Two reinforcement-learning agents training side by side on your local
> GPU, fully visualised in 3D — watch policies evolve from random
> flailing to coherent behaviour in real time.

---

## What this demo is

DEEPRL runs **two independent PPO trainers** in one container, both
visualised in your browser:

1. **Soccer Arena** — agents learn 1v1 or 2v2 soccer from scratch via
   self-play.
2. **Drone Course** — a quadrotor learns to fly under realistic physics,
   advancing through a 15-level curriculum from stable hover to a
   contested combat course.

Each scenario runs **256 parallel environments on the GPU**, generating
thousands of agent–environment interactions per second. The frontend
renders a few of them live in 3D using Three.js while the rest train
silently in the background. Hyperparameters, reward weights, training
speed, curriculum level, and team size are all adjustable from the UI
without restarting the container.

There is **no human-programmed strategy** anywhere in this demo. Every
behaviour you see — positioning, passing, hovering, gate-racing, dodging
projectiles — was discovered by PPO through trial and error.

### Scenario 1 — Soccer Arena

<img width="1036" height="496" alt="deeprl1" src="https://github.com/user-attachments/assets/045bafe4-32c7-4b7a-a8be-90314cecf586" />

A vectorized 2D soccer pitch (rendered in 3D) where the same neural
policy controls both teams (self-play). The default starts at **1v1**;
flip to **2v2** in the UI at any time. 256 matches play in parallel,
and a few are streamed to the browser so you can watch.

What to look for as training progresses:

- **First few minutes:** random kicks, players miss the ball, scores stay
  near zero.
- **Tens of minutes:** agents start chasing the ball deliberately; first
  shots on goal appear.
- **Longer runs:** positioning emerges — one agent presses while the
  other holds back, passes happen between teammates in 2v2,
  goalkeeping behaviour develops.

You can save named checkpoints, reload a snapshot to compare older vs.
newer policies, and tweak reward weights live to see how shaping affects
emergent strategy.

### Scenario 2 — Drone Course

<img width="1039" height="496" alt="deeprl2" src="https://github.com/user-attachments/assets/2d670d0a-2dad-41d7-8329-3e9f373f6b36" />

A 250mm X-config racing quadrotor learns to fly under realistic
**Newton-Euler rigid-body physics** (parameters scaled from documented
Crazyflie 2.x values) with a **MIL-F-8785C Dryden turbulence wind
model**, motor lag, aerodynamic drag, and rotor inertia. The policy
outputs four motor signals; nothing else is hand-tuned.

Training follows a **15-level curriculum** organised in five phases:

| Phase | Levels | Skills the agent learns |
|---|---|---|
| **A — Learn to Fly** | 1–6 | Hover (calm/wind), takeoff (calm/wind), land (calm/wind) |
| **A2 — Intermediate** | 7–9 | Altitude change between cylinders, yaw control, fly-to-point |
| **B — Navigate** | 10–11 | 3-gate waypoint courses (calm, then wind) |
| **C — Precision** | 12–13 | 5-gate courses with obstacles, then add wind + thermals |
| **D — Dynamic Threats** | 14 | 7-gate combat course with moving walls and projectile-firing turrets |
| **E — Contested** | 15 | Final course: all threats + electronic warfare + heavy wind |

Beyond the scripted curriculum there is also:

- **Endless mode** — randomly generated courses for stress-testing.
- **Manual flight** — fly the drone yourself with throttle/pitch/roll/yaw
  controls; record demos for behavior-cloning pretraining.
- **User weapon mode** — shoot projectiles at the drone to see how it
  reacts.
- **Adversary drones** — spawn a hostile red drone (lethal or
  non-lethal) and watch the trained policy respond.
- **Recording & replay** — capture training sessions and replay them
  later.

---

## Capabilities (at a glance)

- Two PPO trainers (soccer + drone) running side by side on one GPU.
- 256 parallel environments per scenario.
- Live 3D viewers (Three.js) with bloom effects for both scenarios.
- Real-time PPO with adjustable hyperparameters and reward weights.
- Self-play (soccer); 15-level curriculum learning (drone).
- Realistic quadrotor physics: Crazyflie-scaled, MIL-F-8785C wind,
  motor lag, aerodynamic drag.
- Manual drone flight + behavior-cloning pretraining from human demos.
- Named checkpoints (save / load / delete) per scenario.
- Endless courses, projectile/EW threats, adversary drones (drone
  scenario).
- Bundled Redis for checkpoint persistence; optional Caddy reverse
  proxy with HTTPS.

---

## Reference build platform

This demo was built and tested on a **Dell Pro Max GB10** (NVIDIA Grace
Blackwell, **ARM / aarch64** architecture). It will run on standard
x86_64 NVIDIA Linux hosts as well, but the bundled Dockerfile pins
PyTorch 2.9.1 + CUDA 13.0 wheels because that's the only stable
combination on aarch64 with the GB10's `sm_121` compute capability. On
older GPUs you may need to change `CUDA_ARCH` (see Configuration below).

---

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Linux | macOS / Windows lack pass-through GPU support — won't work. |
| Docker | 24.x or newer | With Compose **v2** (`docker compose`, not `docker-compose`). |
| GPU | NVIDIA, ≥ 8 GB VRAM | Both trainers + visualiser run together. |
| GPU driver | Recent enough for your CUDA version | `nvidia-smi` must work on the host. |
| NVIDIA Container Toolkit | Installed and configured for Docker | Required to expose the GPU to the container. |
| Disk | ~5 GB | Image (~3 GB) + checkpoints + Redis volume. |
| RAM | 16 GB recommended | 8 GB will work but may swap during initial build. |
| API key | None | Everything runs locally. |

---

## Installation (step-by-step)

These instructions assume a fresh Linux box. If you already have Docker
+ the NVIDIA Container Toolkit working, skip to step 4.

### 1. Verify your GPU is visible to the host

```bash
nvidia-smi
```

You should see a table with your GPU model, driver version, and CUDA
version. If this command fails, **fix your NVIDIA driver before going
further** — the rest will not work.

### 2. Install Docker Engine + Compose v2

Ubuntu / Debian:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"   # let your user run docker without sudo
newgrp docker                      # apply the new group in this shell
docker compose version             # should print "Docker Compose version v2.x.x"
```

If `docker compose version` reports "command not found", install the
plugin:

```bash
sudo apt install docker-compose-plugin
```

### 3. Install the NVIDIA Container Toolkit

Ubuntu / Debian:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify it works inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

You should see the same `nvidia-smi` table you saw on the host. If
this fails, fix it before continuing.

### 4. Clone the repo

```bash
git clone https://github.com/gleaven/deeprl.git
cd deeprl
```

### 5. Create the environment file

```bash
cp .env.example .env
```

The defaults are sensible. Edit `.env` only if you need to change ports
or scale down for a smaller GPU. Common edits:

| Variable | Default | When to change |
|---|---|---|
| `APP_PORT` | `8080` | Port `8080` is already taken on your host. |
| `N_ENVS` | `256` | You have a smaller GPU (try `128` or `64`). |
| `TRAINING_SPEED` | `1` | You want to fast-forward the visualisation. |

### 6. Build and start

```bash
docker compose up -d --build
```

The first build takes **5–15 minutes** (downloads CUDA base image
~3 GB, installs PyTorch + dependencies). Subsequent starts take ~10
seconds.

### 7. Verify it's healthy

```bash
docker compose ps
# both `demo-deeprl` and `demo-redis` should show "healthy" within ~2 min

curl -s http://localhost:8080/health | python3 -m json.tool
```

Expected output (after warm-up):

```json
{
  "status": "ok",
  "service": "deeprl",
  "scenarios": {
    "soccer": {"training": true, "generation": 1, "device": "cuda:0"},
    "drone":  {"training": true, "generation": 1, "device": "cuda:0"}
  }
}
```

### 8. Open the UIs

- **Soccer Arena:** <http://localhost:8080/>
- **Drone Course:** <http://localhost:8080/drone/>

Open them in two browser tabs. Both render live training at ~20 Hz.

### 9. (Optional) Tail the logs

```bash
docker compose logs -f deeprl
```

You should see lines like:

```
DEEPRL ready: soccer=256 envs (cuda:0), drone=256 envs (cuda:0)
PPO [soccer] started: 256 envs, 1v1 on cuda:0
PPO [drone] started: 256 envs on cuda:0
```

---

## Configuration

All variables can be set in `.env` or exported in your shell.

| Variable | Default | What it controls |
|---|---|---|
| `APP_PORT` | `8080` | Browser-facing port for both UIs and the API. |
| `N_ENVS` | `256` | Parallel training envs per scenario. Lower for small GPUs. |
| `TRAINING_SPEED` | `1` | Visualisation playback multiplier (training itself runs as fast as the GPU allows). |
| `REDIS_HOST_PORT` | `6379` | Where the bundled Redis is exposed on the host. |
| `REDIS_URL` | `redis://demo-redis:6379/10` | Connection string used by the trainer; override when you BYO Redis. |
| `DEMO_HOSTNAME` | `localhost` | Hostname Caddy serves under (proxy profile only). |
| `HTTP_PORT` | `8081` | Caddy HTTP port. |
| `HTTPS_PORT` | `8443` | Caddy HTTPS port. |

### Build-time arguments

If you're not on a GB10 / `sm_121` GPU, override the CUDA arch when
building:

```bash
docker compose build --build-arg CUDA_ARCH=8.6   # e.g. RTX 3090
docker compose up -d
```

Common values: `8.0` (A100), `8.6` (RTX 30xx), `8.9` (RTX 40xx),
`9.0` (H100), `12.0`/`12.1` (Grace Blackwell / GB10).

---

## Live controls (in the browser)

Both UIs expose the same core controls:

- **Start / Stop training** — pause learning without losing the policy.
- **Reset** — wipe the policy and start over from random weights.
- **Speed** — visualisation playback multiplier (1× to 16×).
- **Hyperparameters & reward weights** — adjust live; takes effect on
  the next PPO update.
- **Checkpoints** — save the current policy by name, load older ones
  in either *watch* mode (no learning) or *train* mode (resume from
  this point).

Soccer-only:

- **Team size** — toggle 1v1 ↔ 2v2.

Drone-only:

- **Curriculum level** — jump to any level 1–15 directly.
- **Endless mode** — generate a fresh random course on every reset.
- **Manual flight** — take the stick yourself (throttle/pitch/roll/yaw).
- **Demo recording → BC pretraining** — record a few minutes of your own
  flying, then pretrain the policy on those demos before PPO starts.
- **Weapon mode** — fire projectiles at the drone.
- **Adversary** — spawn a red hostile drone (lethal or non-lethal).
- **Recording & replay** — capture training to disk, replay later.

All of these are also exposed as REST + WebSocket API endpoints — open
the network panel in DevTools to see them.

---

## External services (BYO)

If you'd rather use your own Redis (e.g. a managed instance), uncomment
`REDIS_URL` in `.env` and start with the BYO override:

```bash
docker compose -f docker-compose.yml -f docker-compose.byo.yml up -d
```

`docker-compose.byo.yml` removes the bundled `redis` service so only
the `deeprl` container runs locally.

| Variable | Example |
|---|---|
| `REDIS_URL` | `redis://redis.example.com:6379/10` |

Redis is used **only** for storing named checkpoints. The demo will
still train if Redis is unreachable — it just logs a warning and
continues without persistence.

---

## Optional HTTPS reverse proxy

Caddy is bundled as an opt-in profile. It auto-provisions Let's
Encrypt certs when `DEMO_HOSTNAME` is a real DNS name pointing at this
host:

```bash
DEMO_HOSTNAME=deeprl.example.com docker compose --profile proxy up -d
```

For local testing keep `DEMO_HOSTNAME=localhost` and Caddy will issue
a self-signed cert.

---

## Authentication

DEEPRL runs **without authentication** by default. For shared
deployments, put one of these in front of it:

- **Caddy basic auth** — add a `basic_auth` block to the Caddyfile.
- **oauth2-proxy in front of Caddy** — for SSO-style auth.
- **Cloudflare Tunnel + Access policies** — easiest if you're already
  on Cloudflare.

---

## Architecture (file map)

| File | Purpose |
|---|---|
| `server.py` | FastAPI app: REST + WebSocket endpoints, lifespan, broadcast loops. |
| `ppo.py` | Single PPO trainer class, parameterised by scenario. Handles training loop, watch mode, manual mode, BC pretraining, checkpoints. |
| `environment.py` | Vectorized soccer environment (NumPy, batched over `N_ENVS`). |
| `drone_environment.py` | Vectorized quadrotor environment with Newton-Euler physics, curriculum, threats. |
| `recorder.py` | Recording + replay for drone training sessions. |
| `static/` | Soccer UI (Three.js + plain JS). |
| `static-drone/` | Drone UI (Three.js + plain JS). |
| `Caddyfile` | Optional reverse-proxy config. |
| `Dockerfile` | CUDA 13.0 base, PyTorch 2.9.1, Python deps. |

---

## Troubleshooting

- **`nvidia-smi` works on host but not in container** — the NVIDIA
  Container Toolkit isn't wired into Docker. Run `sudo nvidia-ctk
  runtime configure --runtime=docker && sudo systemctl restart docker`
  and try the test container in step 3 again.
- **`Out of memory` on startup** — lower `N_ENVS` (try `128` or `64`).
  256 envs across two scenarios needs roughly 6–8 GB VRAM.
- **`unsupported gpu architecture` during PyTorch import** — your GPU's
  compute capability isn't in the wheel's arch list. Rebuild with
  `--build-arg CUDA_ARCH=<your arch>` (see Configuration).
- **Browser tab freezes / fans spin** — the live 3D viewer is
  GPU-intensive on the *client* side; close other tabs or reduce
  `TRAINING_SPEED`.
- **Port already in use** — change `APP_PORT` in `.env`.
- **`demo-deeprl` health check failing** — give it longer; the first
  PPO generation can take ~60 s on a cold start. Check
  `docker compose logs deeprl` for stack traces.
- **Container restarts in a loop** — almost always a GPU/driver
  mismatch. Confirm `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi`
  works from your shell.
- **Checkpoints disappear after `docker compose down -v`** — `-v`
  removes the named Redis volume. Drop the `-v` to keep them.

---

## FAQ

**Q: Can I use a CPU?** No. PPO on 256 parallel envs is GPU-bound; the
demo will refuse to start without one.

**Q: How long does it take to learn good soccer / drone behaviour?**
Soccer: emergent passing in tens of minutes on a modern GPU. Drone
curriculum: hover in minutes; full curriculum to level 15 in hours.

**Q: Do the two trainers share a GPU cleanly?** Yes — they run as two
processes against the same CUDA device, each with its own optimizer.

**Q: Can I disable one of the scenarios?** Not via env vars currently.
The simplest patch is to comment out the trainer init in `server.py`.

---

## Credits

Built by Andrew Meinecke.

## Components & Licensing

This demo is released under Apache License 2.0. It bundles or wraps
the following third-party components, each retaining its own license:

| Component | License | Use in this demo |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) 2.9.1 + CUDA 13 | BSD-3 | Policy / value network, GPU autodiff for PPO updates |
| [NumPy](https://github.com/numpy/numpy) | BSD-3 | Vectorised environment math (256 parallel worlds) |
| [FastAPI](https://github.com/fastapi/fastapi) | MIT | HTTP + WebSocket server |
| [Uvicorn](https://github.com/encode/uvicorn) | BSD-3 | ASGI server |
| [Redis](https://github.com/redis/redis) (bundled `redis:7-alpine`) | RSALv2 / SSPLv1 (dual) | Training-state persistence (checkpoints, metrics, episode history) |
| [Three.js](https://github.com/mrdoob/three.js/) (bundled in `static/` and `static-drone/`) | MIT | 3D soccer arena + drone-course viewers |
| [NVIDIA CUDA base image](https://hub.docker.com/r/nvidia/cuda) (`nvidia/cuda:13.0.0-devel-ubuntu22.04`) | NVIDIA Deep Learning Container Software License | GPU runtime |

The PPO implementation (`ppo.py`), soccer arena (`environment.py`),
drone course (`drone_environment.py`), and replay recorder
(`recorder.py`) are written from scratch for this demo with no
external RL-library dependency.

### License notes

Redis 7.4+ uses the dual RSALv2 / SSPLv1 license; the RSALv2 path
covers normal use. The NVIDIA CUDA base image is governed by NVIDIA's
Deep Learning Container Software License — read it before
redistributing the built image.
