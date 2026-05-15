# DEEPRL — Live PPO: Soccer Arena + Drone Course

> Two reinforcement-learning scenarios training side by side on your
> local GPU, fully visualised in 3D.

## Overview

A live demonstration of deep reinforcement learning. Two independent
PPO trainers run side by side — one learning **2v2 soccer**, one
learning **quadrotor flight** — each with 256 parallel environments on
the GPU. Watch them in 3D, in real time, as policies evolve from random
flailing to coherent behaviour. All discovered through trial and error
with no human-programmed strategy. Adjust hyperparameters,
reward-shaping weights, and training speed through live controls.

### Scenario 1 — Soccer Arena (2v2)

Agents learn to play 2v2 soccer from scratch using self-play (the same
policy controls both teams). 256 parallel matches generate thousands of
experiences per second. Emergent behaviours appear over training:
positioning, passing, goalkeeping, team coordination — none of it
hand-coded.

### Scenario 2 — Drone Course (Quadrotor Flight)

A 250mm X-config racing quadrotor learns to fly under realistic
Newton-Euler physics (parameters scaled from Crazyflie 2.x; MIL-F-8785C
Dryden turbulence wind model). A curriculum of progressively harder
levels takes the policy from **stable hover → takeoff → landing →
altitude / yaw control → fly-to-waypoint → gate-racing courses**. Motor
dynamics, aerodynamic drag, and rotor inertia are all simulated, so the
learned controller has to deal with the same kind of disturbances a
real quad would face.

## Capabilities

- Two PPO trainers (soccer + drone) running side by side on one GPU.
- 256 parallel environments per scenario.
- Live 3D viewers (Three.js) with bloom effects for both scenarios.
- Real-time PPO with adjustable hyperparameters and reward weights.
- Self-play for soccer; curriculum learning for drone.
- Realistic quadrotor physics (Crazyflie-scaled, with wind).

## Requirements

| Requirement | Notes |
|---|---|
| Docker + Compose v2 | Linux preferred; macOS/Windows lacks GPU support. |
| GPU | NVIDIA GPU + `nvidia-container-toolkit` (required). |
| Disk | ~5 GB for image + model checkpoints. |
| RAM | 16 GB recommended. |
| API key | None required. |

Built and tested on a Dell Pro Max GB10 (ARM architecture).

## Quick Start

```bash
git clone <repo-url> && cd deeprl
cp .env.example .env
docker compose up -d --build
open http://localhost:${APP_PORT:-8080}
```

## Configuration

| Variable | Default | What it controls |
|---|---|---|
| `APP_PORT` | `8080` | Browser-facing port. |
| `N_ENVS` | `256` | Parallel training envs (lower for small GPUs). |
| `TRAINING_SPEED` | `1` | Visualisation playback multiplier. |

## External Services (BYO)

```bash
docker compose -f docker-compose.yml -f docker-compose.byo.yml up -d
```

| Variable | Example |
|---|---|
| `REDIS_URL` | `redis://redis.example.com:6379/10` |

## Optional HTTPS Proxy

```bash
DEMO_HOSTNAME=deeprl.example.com docker compose --profile proxy up -d
```

## Auth Notes

Runs **without authentication** by default. Add a layer for shared
deployments: Caddy basic auth, oauth2-proxy in front of Caddy, or a
Cloudflare Tunnel with Access policies.

## Troubleshooting

- **GPU not visible:** confirm `nvidia-smi` works on the host and
  `nvidia-container-toolkit` is installed.
- **Out of memory at start:** lower `N_ENVS` (e.g. 64 or 128) in
  `.env`.
- **Browser tab freezes:** the live 3D viewer is GPU-intensive; close
  other tabs or reduce `TRAINING_SPEED`.
- **Port collision:** change `APP_PORT` in `.env`.

## Credits

Built by Andrew Meinecke.
