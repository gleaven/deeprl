# DEEPRL — Live PPO 2v2 Soccer

> Watch 256 parallel matches train simultaneously on your local GPU.
> Agents learn 2v2 soccer from scratch with PPO, fully visualised in 3D.

## Overview

A live demonstration of deep reinforcement learning where AI agents
learn to play 2v2 soccer entirely from scratch. 256 parallel matches
train simultaneously on the GPU using Proximal Policy Optimization
(PPO), generating thousands of experiences per second. Watch the 3D
arena in real time as agents evolve from random flailing to emergent
behaviours — positioning, passing, goalkeeping, team coordination — all
discovered through trial and error with no human-programmed strategy.
Adjust hyperparameters, reward-shaping weights, and training speed
through live controls.

## Capabilities

- 256 parallel environments on a single GPU.
- Live 3D arena (Three.js) with bloom effects.
- Real-time PPO with adjustable hyperparameters.
- Self-play: same policy plays both teams.

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
