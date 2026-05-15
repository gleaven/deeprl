# DEEPRL -- Autonomous Drone & Soccer RL -- Demo Guide

## Overview

DEEPRL is a dual reinforcement learning demonstration running two independent training scenarios simultaneously on the GB10 GPU:

1. **Soccer (2v2 Self-Play)**: AI agents learn to play 2v2 soccer entirely from scratch. Same neural network controls both teams (self-play). Agents evolve from random movement to emergent team behaviors -- positioning, passing, goalkeeping -- all discovered through trial and error.

2. **DroneRL (Autonomous Quadrotor)**: A simulated 250mm racing quadrotor learns to navigate a 3D obstacle course while handling kinetic threats (projectiles), electronic warfare (GPS denial, sensor jamming), and environmental disturbances (wind, thermals, gusts). The physics model is based on published research (Crazyflie 2.x, gym-pybullet-drones) and designed for sim-to-real transfer.

Both scenarios use Proximal Policy Optimization (PPO) with 64 parallel environments, running on the same GPU with shared infrastructure. Training is live and continuous -- you watch the agents learn in real time through interactive Three.js visualizations.

## Architecture

### System Components

- **Container**: `demo-deeprl` -- single FastAPI application hosting both trainers
- **Port**: 8080 (internal), exposed via Traefik at `/deeprl`
- **Dependencies**: Redis DB 10 (checkpoints, metrics), CUDA GPU
- **Training Engine**: Two `PPOTrainer` threads running concurrently (soccer + drone)
- **Physics**: Pure NumPy vectorized environments (no external physics engine)
- **Visualization**: Three.js with UnrealBloomPass post-processing
- **WebSocket**: Real-time state broadcast at ~10Hz per scenario

### Data Flow

```
PPOTrainer Thread (Soccer)                PPOTrainer Thread (Drone)
         |                                         |
   VectorizedSoccerEnv                    VectorizedDroneEnv
   (64 parallel matches)                  (64 parallel courses)
         |                                         |
   step() -> obs, reward, done            step() -> obs, reward, done
         |                                         |
   PPO update (GPU)                       PPO update (GPU)
         |                                         |
   get_display_state(env=0)               get_display_state(env=0)
         |                                         |
   WebSocket /ws/arena                    WebSocket /ws/course
         |                                         |
   Three.js Soccer Arena                  Three.js Drone Course
   (browser at /deeprl)                   (browser at /deeprl/drone)
```

### AI/ML Models

| Component | Details |
|-----------|---------|
| Algorithm | Proximal Policy Optimization (PPO) with clipped surrogate objective |
| Network | Shared MLP backbone (256-256) with separate actor (policy) and critic (value) heads |
| Actor output | Continuous actions via Normal distribution (mean + log_std per action dim) |
| Framework | PyTorch 2.9.1 + CUDA 13.0 on GB10 Blackwell GPU |
| Training rate | ~12,000 environment steps/second across 64 parallel envs |

### Drone Physics Model

The quadrotor simulation implements Newton-Euler rigid body dynamics based on published research:

| Parameter | Value | Source |
|-----------|-------|--------|
| Mass | 0.5 kg | 250mm racing quad typical |
| Arm length | 0.125 m | Center to motor |
| Prop radius | 0.065 m | 5-inch propeller |
| Thrust coefficient (KF) | 1.82e-6 N/(rad/s)^2 | Crazyflie kf scaled by R^4, converted RPM->rad/s |
| Torque coefficient (KM) | 4.56e-8 N*m/(rad/s)^2 | Crazyflie km scaled, converted RPM->rad/s |
| Max RPM | 12,000 | 2300KV motor on 4S LiPo |
| Hover RPM | ~7,838 | 65% throttle (T/W ratio 2.34) |
| Physics rate | 200 Hz | 4 substeps per RL step |
| RL decision rate | 50 Hz | Real-time deployable |

**Physics pipeline per substep:**
1. Motor lag (first-order response, tau=20ms)
2. Thrust from blade element theory (F = KF * omega^2)
3. Ground effect (Cheeseman & Bennett 1955: thrust boost near surface)
4. X-config torque mixing (roll/pitch/yaw from 4 motors)
5. Gyroscopic precession from spinning rotors
6. Euler's rotation equation (I*alpha = tau - omega x I*omega)
7. Quaternion integration with renormalization
8. Translational forces (thrust + gravity + aerodynamic drag + wind)
9. Ground normal force (hard constraint, no penetration)
10. Semi-implicit Euler integration
11. Ground collision (inelastic, friction damping)

**References:**
- Luukkonen (2011) "Modelling and control of quadcopter"
- Panerati et al. (2021) gym-pybullet-drones
- MIT Lecture 6: Quadrotor Dynamics
- MIL-F-8785C Flying Qualities (Dryden wind model)
- Cheeseman & Bennett (1955) ground effect model

### Observation Space (Drone: 46 dimensions)

Every observation corresponds to a sensor available on real hardware. No GPS position -- forces inertial navigation.

| Index | Dims | Sensor | Description |
|-------|------|--------|-------------|
| 0-2 | 3 | IMU accelerometer | Body-frame acceleration (includes gravity) |
| 3-5 | 3 | IMU gyroscope | Body-frame angular velocity |
| 6-9 | 4 | AHRS fusion | Orientation quaternion [w,x,y,z] |
| 10-12 | 3 | Dead reckoning | Velocity in body frame |
| 13 | 1 | Barometer | Height above ground |
| 14 | 1 | Baro derivative | Vertical velocity |
| 15-18 | 4 | ESC telemetry | Normalized motor RPMs |
| 19-30 | 12 | LIDAR array | 12 distance readings (10m max range) |
| 31-33 | 3 | Waypoint bearing | Direction to next gate (body frame) |
| 34 | 1 | Waypoint range | Distance to next gate |
| 35 | 1 | Mission progress | Waypoint completion fraction |
| 36 | 1 | Clock | Time remaining fraction |
| 37-39 | 3 | Accel residuals | Wind estimate (body frame) |
| 40 | 1 | GPS receiver | GPS available flag |
| 41 | 1 | Sensor health | BITE fraction |
| 42 | 1 | Battery | Charge remaining |
| 43-45 | 3 | Threat detection | Bearing to nearest threat |

### Action Space (Drone: 4 dimensions)

Raw motor commands: 4 values in [-1, 1] mapped to [0, MAX_OMEGA]. No PID controller -- the neural network *is* the controller. It learns the full mixing matrix (collective thrust + roll/pitch/yaw torques) implicitly. This is the most sim-to-real transferable action space.

### Threat & Environment Systems

| System | Details |
|--------|---------|
| **Wind** | MIL-F-8785C Dryden turbulence (first-order discrete filter), thermal updrafts near fire zones, random gusts (3-8 m/s, 0.5-2s duration) |
| **Projectiles** | Up to 5 simultaneous, 15 m/s with ballistic trajectory, spawned from course perimeter with lead prediction |
| **Electronic Warfare** | GPS denial zones (bearing/range zeroed), sensor jamming (Gaussian noise injection), random sensor dropout |
| **Battery** | 300s capacity at hover, drain proportional to motor power (kf * omega^3) |

### Curriculum System (15 Levels)

Baby-step progression: stability → precision → speed. Each skill is introduced in calm, then wind.

| Lvl | Name | Task | Gates | Obstacles | Wind | Threats | EW |
|-----|------|------|-------|-----------|------|---------|-----|
| 1 | Hover (Calm) | Hold altitude 4s | None | None | None | None | None |
| 2 | Hover (Wind) | Hold altitude 4s | None | None | Light 0-2 m/s | None | None |
| 3 | Takeoff (Calm) | Launch from ground to 3m | None | None | None | None | None |
| 4 | Takeoff (Wind) | Launch from ground to 3m | None | None | Light 0-2 m/s | None | None |
| 5 | Land (Calm) | Gentle touchdown from 5m | None | None | None | None | None |
| 6 | Land (Wind) | Gentle touchdown from 5m | None | None | Light 0-2 m/s | None | None |
| 7 | Altitude Change | Climb/descend to 3 targets | None | None | Very light | None | None |
| 8 | Yaw Control | Rotate to 3 heading targets | None | None | Very light | None | None |
| 9 | Fly to Point | Navigate to target position | None | None | Light 0-1 m/s | None | None |
| 10 | Waypoints (Calm) | 3 wide gates | 3 wide | None | Very light | None | None |
| 11 | Waypoints (Wind) | 3 wide gates | 3 wide | None | Moderate 0-3 m/s | None | None |
| 12 | Obstacles (Calm) | 5 gates + obstacles | 5 | Columns + walls | Light 0-1.5 m/s | None | None |
| 13 | Obstacles (Wind) | 5 gates + obstacles | 5 | Columns + walls | Moderate + thermals | None | None |
| 14 | Combat Course | 7 gates + threats | 7 | Static + moving | Strong 0-4 m/s | Projectiles | None |
| 15 | Final Course | 9 narrow gates, all threats | 9 narrow | All types | Severe 0-5 m/s | Heavy | Full EW |

Each level has its own reward weight profile (stability-heavy early, speed/dodge later). Auto-unlock: 70% completion rate over 100 episodes advances to the next level.

### Domain Randomization (Sim-to-Real)

Per-episode randomization produces robust, transferable policies:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Mass | 0.4-0.65 kg | Battery/payload variation |
| Inertia (Ixx, Iyy) | +/-20% | Frame variation |
| Thrust coeff (KF) | +/-20% | Prop wear, air density |
| Motor time constant | 10-40 ms | ESC response variation |
| Drag coefficients | +/-50% | Airframe variation |
| Turbulence intensity | 0-4.5 m/s | Calm to severe weather |

### Database / State

| Store | Purpose |
|-------|---------|
| Redis DB 10 | Checkpoints (auto-save every 10 gens), named checkpoints, training metrics |
| In-memory | All environment state (positions, velocities, quaternions, obstacles, projectiles) |

## User Interface

### Soccer View (`/deeprl`)

- **3D Arena**: Three.js pitch with player capsules (cyan vs magenta), ball, goals, boundary walls
- **Training Curves**: Real-time charts for reward, policy loss, value loss, entropy
- **Controls**: Start/pause training, hyperparameter sliders (LR, entropy), reward weight sliders
- **Checkpoint Manager**: Save, load, delete named checkpoints
- **Speed Control**: 1x-50x training speed multiplier

### Drone View (`/deeprl/drone`)

- **3D Course**: Three.js obstacle course with walls, columns, gates, moving obstacles
- **Drone Model**: Central body with 4 arms, spinning propellers, guard rings, LED lights
- **LIDAR Visualization**: 12 rays from drone with green-to-red distance gradient
- **Effects**: Projectile trails, wind particles, thermal columns, EW domes
- **HUD**: Attitude indicator, altitude bar, 4 motor RPM gauges, battery, EW status
- **Training Curves**: Reward, waypoint rate, collision rate
- **Controls**: Curriculum level selector, reward/penalty sliders, hyperparameter tuning
- **Camera**: Orbit mode with auto-follow

### Visual Theme

- **Soccer**: Cyan (`#00e5ff`) accent, dark arena, bloom lighting
- **Drone**: Amber/orange (`#ff9900`) military theme, dark ground, grid overlay

## User Stories

1. **As a technical audience member**, I want to see neural networks learn a complex task from scratch in real time, so I can understand how reinforcement learning discovers strategies through trial and error.

2. **As an AI researcher**, I want to adjust hyperparameters and reward weights live, so I can demonstrate how reward shaping and training dynamics affect emergent behavior.

3. **As a defense/aerospace professional**, I want to see a sim-to-real drone controller learning to navigate under adversarial conditions (wind, projectiles, EW), so I can evaluate edge-deployed RL for autonomous systems.

4. **As an executive**, I want to see a visually impressive demonstration of AI learning in real time on a single edge device, so I can understand the capability of the GB10 for compute-intensive workloads.

5. **As a robotics engineer**, I want to see physically accurate quadrotor dynamics with domain randomization, so I can evaluate the sim-to-real transfer potential of the trained policy.

## Demo Walkthrough

### Soccer Flow

1. **Open DEEPRL**: Navigate to `/deeprl`. The 3D arena loads with two teams of players and a ball.

2. **Explain the Setup**: "These AI agents started with zero knowledge of soccer. No rules, no strategy -- just a neural network receiving sensor data and outputting movement commands. They learn entirely through trial and error."

3. **Watch Early Training**: If starting from scratch, agents flail randomly. Point out the reward curve climbing as they discover that approaching the ball is rewarded.

4. **Show Emergent Behaviors**: After training, agents exhibit positioning, ball pursuit, basic passing, and even goalkeeping -- all self-discovered. "Nobody programmed 'go to the ball' or 'guard the goal' -- these strategies emerged from the reward signal."

5. **Adjust Rewards Live**: Change a reward weight (e.g., increase kick_reward). Watch how behavior shifts in real time. "We can shape what the AI values, and it adapts its strategy."

6. **Show Scale**: "64 matches are happening simultaneously on the GPU right now. Each match generates training data. In a few minutes, these agents play more soccer than a human could in years."

### Drone Flow

7. **Switch to Drone**: Click the "DRONE" tab or navigate to `/deeprl/drone`. The obstacle course loads with walls, columns, gates, and the drone model.

8. **Explain the Physics**: "This isn't a game -- it's a real physics simulation. Newton-Euler rigid body dynamics, quaternion rotation, aerodynamic drag, motor lag. The same equations used in flight controllers like PX4. The action space is raw motor RPM -- the neural network learns to be a flight controller."

9. **Watch the Learning**: Early on, the drone crashes immediately. Over generations, it learns to hover, then navigate, then dodge projectiles. "It's discovering flight from first principles."

10. **Show the Obstacle Course**: Point out walls, gates (green = next target), moving obstacles. "The curriculum system automatically increases difficulty as the drone masters each level."

11. **Demonstrate Threats**: At curriculum level 14, projectiles appear. "Now it has to navigate AND dodge incoming fire. At level 15, we add GPS denial and sensor jamming -- the drone has to fly blind using only IMU and LIDAR."

12. **Show the Wind**: Point out wind particles and thermal columns. "This is a Dryden turbulence model -- the same military standard used for flight simulation. Random gusts, thermals, crosswinds."

13. **Adjust Training**: Show the reward sliders. Increase collision penalty to make it more cautious. Decrease stability penalty to allow aggressive maneuvers. "We're shaping the AI's priorities in real time."

14. **The Edge Story**: "All of this -- 64 parallel simulations, neural network training, physics computation -- runs on this single device. A drone could carry this and retrain its flight controller in the field."

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Soccer HTML page |
| GET | `/drone/` | Drone HTML page |
| GET | `/health` | Health check |
| WebSocket | `/ws/arena` | Soccer real-time state (init, state_update, training_update) |
| WebSocket | `/ws/course` | Drone real-time state (init, course_layout, state_update, training_update) |

### WebSocket Commands (sent by client)

| Command | Payload | Description |
|---------|---------|-------------|
| `start_training` | -- | Start/resume PPO training |
| `stop_training` | -- | Pause training |
| `set_config` | `{key: value}` | Update hyperparameters or reward weights live |
| `set_curriculum` | `{level: 1-15}` | Set drone curriculum level |
| `save_checkpoint` | `{name: "..."}` | Save named checkpoint |
| `load_checkpoint` | `{name: "...", mode: "train"\|"watch"}` | Load checkpoint |
| `delete_checkpoint` | `{name: "..."}` | Delete checkpoint |
| `set_speed` | `{speed: 1-50}` | Training speed multiplier |

## Capabilities Demonstrated

- **Live deep reinforcement learning** on GB10 GPU (PPO with continuous action spaces)
- **64 parallel environments** for massively parallel data collection
- **Real-time 3D visualization** with Three.js and WebSocket streaming
- **Physically accurate quadrotor simulation** (Newton-Euler, quaternions, blade element theory)
- **Sim-to-real design**: raw motor actions, real sensor observations, domain randomization
- **Adversarial conditions**: kinetic threats, electronic warfare, environmental disturbances
- **Military-standard wind model** (MIL-F-8785C Dryden turbulence)
- **Ground effect modeling** (Cheeseman & Bennett 1955)
- **Curriculum learning** with automatic difficulty progression
- **Live hyperparameter tuning** and reward shaping
- **Checkpoint management** for saving and replaying trained policies
- **Dual-scenario training** on a single GPU (soccer + drone simultaneously)
- **Self-play** in soccer for open-ended emergent behavior

## Desired Outcomes

- **Primary**: Demonstrate that a single edge device can run computationally intensive reinforcement learning training in real time, producing capable autonomous agents.
- **Secondary**: Show that physically accurate drone simulation with adversarial conditions can run on the GB10, producing policies designed for sim-to-real transfer.
- **Executive**: Illustrate the concept of autonomous systems that learn and adapt at the edge -- no cloud dependency, no pre-programmed behaviors.
- **Technical**: Prove that PPO with domain randomization and curriculum learning can produce robust drone controllers from raw motor commands.

## Technical Notes

- Route: `/deeprl` (soccer), `/deeprl/drone` (drone)
- Container: `demo-deeprl`
- Port: 8080
- Redis: DB 10 (checkpoints + metrics)
- Soccer accent: `#00e5ff` (Cyan)
- Drone accent: `#ff9900` (Amber)
- GPU: CUDA 13.0, PyTorch 2.9.1
- Physics: 200 Hz (drone), 100 Hz (soccer)
- RL rate: 50 Hz (drone), 20 Hz (soccer)
- Broadcast rate: ~10 Hz (both)
- Checkpoint auto-save: every 10 generations
- Training starts automatically on container startup
- Both trainers share one GPU via PyTorch's default CUDA context
