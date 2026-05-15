"""Vectorized Quadrotor Drone Environment with Real Physics.

Physics based on Newton-Euler rigid body dynamics for a 250mm X-config racing quad.
Parameters scaled from documented Crazyflie 2.x values (gym-pybullet-drones, Forster 2015).
Wind model based on MIL-F-8785C Dryden turbulence specification.

References:
  - Luukkonen (2011) "Modelling and control of quadcopter"
  - Panerati et al. (2021) "Learning to Fly" / gym-pybullet-drones
  - MIT Lecture 6: Quadrotor Dynamics (vnav.mit.edu)
  - MIL-F-8785C Flying Qualities of Piloted Airplanes
"""

import numpy as np

# ── Quadrotor Physical Constants (250mm X-config) ─────────────
# Scaled from Crazyflie 2.x: arm_ratio=3.15x, mass_ratio=18.5x
MASS            = 0.5           # kg
ARM_LENGTH      = 0.125         # m (center to motor)
PROP_RADIUS     = 0.065         # m (5-inch prop)
DRONE_RADIUS    = 0.15          # m (collision sphere)

# Inertia tensor (diagonal, symmetric X-frame)
# I_scaled = I_cf * mass_ratio * length_ratio^2
IXX             = 2.32e-3       # kg*m^2
IYY             = 2.32e-3       # kg*m^2
IZZ             = 4.00e-3       # kg*m^2
J_MOTOR         = 3.5e-5        # kg*m^2 (rotor moment of inertia)

# Motor thrust: F = KF * omega^2 (blade element theory)
# Crazyflie kf_rpm=3.16e-10 N/RPM^2 at prop_r=0.023m, scales as R^4 → 2.02e-8 N/RPM^2
# Convert RPM→rad/s: KF_rads = KF_rpm * (60/(2*pi))^2 ≈ KF_rpm * 91.19
KF              = 1.82e-6       # N/(rad/s)^2  (2.0e-8 * 91.19)
KM              = 4.56e-8       # N*m/(rad/s)^2 (5.0e-10 * 91.19)
MAX_RPM         = 12000.0       # RPM (2300KV motor on 4S)
GRAVITY         = 9.81          # m/s^2

MAX_OMEGA       = MAX_RPM * 2.0 * np.pi / 60.0   # ~1257 rad/s
HOVER_OMEGA     = np.sqrt(MASS * GRAVITY / (4.0 * KF))  # ~7800 RPM equivalent

# Aerodynamic drag (linear model, adequate for <10 m/s)
DRAG_XY         = 0.01          # N/(m/s)
DRAG_Z          = 0.012         # N/(m/s)
DRAG_ROT        = 0.0005        # N*m/(rad/s)

# Motor dynamics (first-order lag)
MOTOR_TAU       = 0.02          # s (brushless ESC response)

# Simulation timing
DT              = 0.005         # s physics substep (200 Hz)
SUBSTEPS        = 4             # substeps per RL step
RL_DT           = DT * SUBSTEPS # 0.02s = 50 Hz RL decision rate

# Effective arm length for X-config torque (45-deg offset)
L_EFF           = ARM_LENGTH / np.sqrt(2.0)

# ── Course Constants ──────────────────────────────────────────
COURSE_W        = 40.0          # m (x-axis)
COURSE_D        = 40.0          # m (y-axis)
COURSE_H        = 20.0          # m (z-axis ceiling)
MAX_LIDAR_RANGE = 10.0          # m

# ── Observation / Action Sizes ────────────────────────────────
OBS_SIZE        = 50
ACTION_SIZE     = 4             # 4 motor RPM commands

# ── Reward Constants ──────────────────────────────────────────
WAYPOINT_REWARD         = 5.0
PROGRESS_REWARD         = 0.05
COURSE_COMPLETE_REWARD  = 50.0
SURVIVAL_REWARD         = 0.01
DODGE_BONUS             = 0.5

COLLISION_PENALTY       = -10.0
PROJECTILE_HIT_PENALTY  = -5.0
OOB_PENALTY             = -10.0
CRASH_PENALTY           = -10.0
STABILITY_COEFF         = 0.01
ENERGY_COEFF            = 0.001
SMOOTHNESS_COEFF        = 0.005   # penalize motor command jerk (delta between steps)
ALTITUDE_COEFF          = 0.005   # reward for staying in target altitude band
ALTITUDE_TARGET_LO      = 2.0    # m — bottom of preferred altitude band
ALTITUDE_TARGET_HI      = 8.0    # m — top of preferred altitude band
ORIENTATION_COEFF       = 0.01   # penalize being inverted (qw near 0)
SPEED_COEFF             = 0.002  # penalize excessive velocity
SPEED_LIMIT             = 10.0   # m/s — penalty kicks in above this
GATE_ALIGN_BONUS        = 1.0    # bonus for perpendicular gate approach
TIMEOUT_PENALTY         = -15.0  # penalty for running out of time on gate courses

# ── Hover / Takeoff / Landing Completion ─────────────────────
HOVER_TARGET_DURATION   = 2.0    # seconds of stable hover to "complete" hover levels
HOVER_POS_RADIUS        = 1.0    # m — max horizontal drift from start for hover hold
HOVER_MAX_TILT          = 0.35   # rad (~20 deg) — max tilt during hover hold
HOVER_SHRINK_STAGES     = 5      # number of times the box shrinks
HOVER_SHRINK_FACTOR     = 0.8    # multiply radius & alt band each stage (0.8^5 ≈ 0.33)
TAKEOFF_TARGET_ALT      = 3.0    # m — target altitude for takeoff completion
TAKEOFF_HOLD_DURATION   = 3.0    # seconds holding altitude after takeoff
LANDING_MAX_VEL         = 0.5    # m/s — max descent speed for gentle landing
LANDING_REWARD          = 10.0   # reward for gentle touchdown
DRIFT_COEFF             = 0.02   # per-step penalty for horizontal drift from start

# ── Altitude Change / Yaw / Fly-to-Point Completion ─────────
ALT_CHANGE_HOLD         = 2.0    # seconds holding at target altitude to complete
ALT_CHANGE_TOLERANCE    = 0.5    # m — acceptable altitude error
ALT_CYL_RADIUS          = 2.0   # m — radius of each altitude cylinder zone
ALT_CYL_HEIGHT          = 2.0   # m — half-height of each cylinder zone
YAW_TARGET_HOLD         = 2.0    # seconds holding target yaw to complete
YAW_TOLERANCE           = 0.15   # rad (~8.6 deg) — acceptable yaw error
YAW_ZONE_RADIUS         = 3.0    # m — containment zone radius (pirouette in place)
FLY_TO_RADIUS           = 1.5    # m — distance to target point for completion
FLY_TO_HOLD             = 1.5    # seconds hovering near target to complete

# ── Curriculum Profiles (15 levels) ──────────────────────────
# Each level defines: max_steps, wind/turb ranges, domain randomization scale,
# feature flags (thermals, projectiles, EW), start/completion mode, and
# per-level reward weights.
# Progression: stability → precision → speed
#
# start_mode: 'air' (start at 2m), 'ground' (start on ground), 'hover' (start at 5m)
# completion_mode: 'hover', 'takeoff', 'land', 'altitude_change', 'yaw',
#                  'fly_to_point', 'gates'
# Shared reward template for flight basics (hover/takeoff/land) levels
_FLIGHT_REWARDS = {
    'survival_reward': 0.05,
    'altitude_coeff': 0.03,
    'stability_coeff': 0.03,
    'orientation_coeff': 0.03,
    'smoothness_coeff': 0.02,
    'drift_coeff': DRIFT_COEFF,
    'energy_coeff': 0.0005,
    'speed_coeff': 0.001,
    'waypoint_reward': 0.0,
    'progress_reward': 0.0,
    'course_complete_reward': 0.0,
    'gate_align_bonus': 0.0,
    'dodge_bonus': 0.0,
    'proj_avoid_bonus': 0.0,
    'collision_penalty': -10.0,
    'crash_penalty': -10.0,
    'oob_penalty': -10.0,
    'projectile_hit_penalty': -5.0,
}
# Intermediate flight skills rewards — similar to flight basics but with navigation hints
_INTERMEDIATE_REWARDS = {
    **_FLIGHT_REWARDS,
    'survival_reward': 0.03,
    'stability_coeff': 0.025,
    'orientation_coeff': 0.025,
    'smoothness_coeff': 0.015,
    'drift_coeff': 0.01,
}

CURRICULUM_PROFILES = {
    # ══ Phase A: Learn to Fly ══════════════════════════════════
    1: {
        'name': 'Hover (Calm)',
        'max_steps': 1250,  # 25s — 4×2s + 10s final hold + recovery margin
        'start_mode': 'air', 'completion_mode': 'hover',
        'hover_stages': 5,  # full shrink progression — learn precision in calm
        'wind_range': (0.0, 0.0), 'turb_range': (0.0, 0.0),
        'dr_scale': 0.3,
        'thermals': False, 'projectiles': False, 'ew': False,
        'rewards': {**_FLIGHT_REWARDS},
    },
    2: {
        'name': 'Hover (Wind)',
        'max_steps': 1000,  # 20s — 2×2s + 10s final hold + recovery margin under wind
        'start_mode': 'air', 'completion_mode': 'hover',
        'hover_stages': 3,  # progressive shrink under wind
        'wind_range': (0.0, 2.0), 'turb_range': (0.5, 1.5),
        'dr_scale': 0.5,
        'thermals': False, 'projectiles': False, 'ew': False,
        'rewards': {**_FLIGHT_REWARDS},
    },
    3: {
        'name': 'Takeoff (Calm)',
        'max_steps': 500,  # 10s — reach 3m alt + hold, plenty of time
        'start_mode': 'ground', 'completion_mode': 'takeoff',
        'wind_range': (0.0, 0.0), 'turb_range': (0.0, 0.0),
        'dr_scale': 0.3,
        'thermals': False, 'projectiles': False, 'ew': False,
        'rewards': {**_FLIGHT_REWARDS, 'drift_coeff': 0.01},
    },
    4: {
        'name': 'Takeoff (Wind)',
        'max_steps': 750,  # 15s — takeoff under wind needs more recovery time
        'start_mode': 'ground', 'completion_mode': 'takeoff',
        'wind_range': (0.0, 2.0), 'turb_range': (0.5, 1.5),
        'dr_scale': 0.5,
        'thermals': False, 'projectiles': False, 'ew': False,
        'rewards': {**_FLIGHT_REWARDS, 'drift_coeff': 0.01},
    },
    5: {
        'name': 'Land (Calm)',
        'max_steps': 500,  # 10s — descend from hover to ground
        'start_mode': 'hover', 'completion_mode': 'land',
        'wind_range': (0.0, 0.0), 'turb_range': (0.0, 0.0),
        'dr_scale': 0.3,
        'thermals': False, 'projectiles': False, 'ew': False,
        'rewards': {**_FLIGHT_REWARDS,
            'altitude_coeff': 0.0,
            'drift_coeff': 0.01,
        },
    },
    6: {
        'name': 'Land (Wind)',
        'max_steps': 750,  # 15s — landing in wind needs caution
        'start_mode': 'hover', 'completion_mode': 'land',
        'wind_range': (0.0, 2.0), 'turb_range': (0.5, 1.5),
        'dr_scale': 0.5,
        'thermals': False, 'projectiles': False, 'ew': False,
        'rewards': {**_FLIGHT_REWARDS,
            'altitude_coeff': 0.0,
            'drift_coeff': 0.01,
        },
    },
    # ══ Phase A2: Intermediate Flight Skills ═══════════════════
    7: {
        'name': 'Altitude Change',
        'max_steps': 1500,  # 30s — fly between 3 cylinders (up to 14m apart) + 2s hold each
        'start_mode': 'air', 'completion_mode': 'altitude_change',
        'wind_range': (0.0, 0.5), 'turb_range': (0.0, 0.3),
        'dr_scale': 0.4,
        'thermals': False, 'projectiles': False, 'ew': False,
        'rewards': {**_INTERMEDIATE_REWARDS, 'altitude_coeff': 0.04},
    },
    8: {
        'name': 'Yaw Control',
        'max_steps': 1000,  # 20s — 3 yaw rotations with hold time between each
        'start_mode': 'air', 'completion_mode': 'yaw',
        'wind_range': (0.0, 0.5), 'turb_range': (0.0, 0.3),
        'dr_scale': 0.4,
        'thermals': False, 'projectiles': False, 'ew': False,
        'rewards': {**_INTERMEDIATE_REWARDS,
            'orientation_coeff': 0.0,   # disable — penalizes yaw rotation via 1-qw²
            'stability_coeff': 0.015,   # reduce — ang_vel_mag includes yaw rate
        },
    },
    9: {
        'name': 'Fly to Point',
        'max_steps': 1000,  # 20s — fly 10-25m + 1.5s hold at target
        'start_mode': 'air', 'completion_mode': 'fly_to_point',
        'wind_range': (0.0, 1.0), 'turb_range': (0.0, 0.5),
        'dr_scale': 0.5,
        'thermals': False, 'projectiles': False, 'ew': False,
        'rewards': {**_INTERMEDIATE_REWARDS,
            'progress_reward': 0.1,
            'drift_coeff': 0.0,
        },
    },
    # ══ Phase B: Learn to Navigate ═════════════════════════════
    10: {
        'name': 'Waypoints (Calm)',
        'max_steps': 1250,  # 25s — 3 gates spread across 28m course
        'start_mode': 'air', 'completion_mode': 'gates',
        'wind_range': (0.0, 0.5), 'turb_range': (0.0, 0.5),
        'dr_scale': 0.5,
        'thermals': False, 'projectiles': False, 'ew': False,
        'rewards': {
            'survival_reward': 0.0,
            'altitude_coeff': 0.01,
            'stability_coeff': 0.02,
            'orientation_coeff': 0.02,
            'smoothness_coeff': 0.015,
            'drift_coeff': 0.0,
            'energy_coeff': 0.001,
            'speed_coeff': 0.002,
            'waypoint_reward': 5.0,
            'progress_reward': 0.1,
            'course_complete_reward': 30.0,
            'gate_align_bonus': 0.5,
            'dodge_bonus': 0.0,
            'timeout_penalty': TIMEOUT_PENALTY,
            'collision_penalty': -10.0,
            'crash_penalty': -10.0,
            'oob_penalty': -10.0,
            'projectile_hit_penalty': -5.0,
        },
    },
    11: {
        'name': 'Waypoints (Wind)',
        'max_steps': 1500,  # 30s — 3 gates under wind, needs correction time
        'start_mode': 'air', 'completion_mode': 'gates',
        'wind_range': (0.0, 3.0), 'turb_range': (0.5, 2.5),
        'dr_scale': 0.7,
        'thermals': False, 'projectiles': False, 'ew': False,
        'rewards': {
            'survival_reward': 0.0,
            'altitude_coeff': 0.01,
            'stability_coeff': 0.02,
            'orientation_coeff': 0.02,
            'smoothness_coeff': 0.015,
            'drift_coeff': 0.0,
            'energy_coeff': 0.001,
            'speed_coeff': 0.002,
            'waypoint_reward': 5.0,
            'progress_reward': 0.1,
            'course_complete_reward': 30.0,
            'gate_align_bonus': 0.5,
            'dodge_bonus': 0.0,
            'timeout_penalty': TIMEOUT_PENALTY,
            'collision_penalty': -10.0,
            'crash_penalty': -10.0,
            'oob_penalty': -10.0,
            'projectile_hit_penalty': -5.0,
        },
    },
    # ══ Phase C: Precision Flying ══════════════════════════════
    12: {
        'name': 'Obstacles (Calm)',
        'max_steps': 1750,  # 35s — 5 gates + obstacle avoidance
        'start_mode': 'air', 'completion_mode': 'gates',
        'wind_range': (0.0, 1.5), 'turb_range': (0.0, 1.0),
        'dr_scale': 0.7,
        'thermals': False, 'projectiles': False, 'ew': False,
        'rewards': {
            'survival_reward': 0.0,
            'altitude_coeff': 0.005,
            'stability_coeff': 0.015,
            'orientation_coeff': 0.015,
            'smoothness_coeff': 0.01,
            'drift_coeff': 0.0,
            'energy_coeff': 0.001,
            'speed_coeff': 0.002,
            'waypoint_reward': 5.0,
            'progress_reward': 0.05,
            'course_complete_reward': 50.0,
            'gate_align_bonus': 1.0,
            'dodge_bonus': 0.0,
            'timeout_penalty': TIMEOUT_PENALTY,
            'collision_penalty': -10.0,
            'crash_penalty': -10.0,
            'oob_penalty': -10.0,
            'projectile_hit_penalty': -5.0,
        },
    },
    13: {
        'name': 'Obstacles (Wind)',
        'max_steps': 2000,  # 40s — 5 gates + obstacles + wind/thermals
        'start_mode': 'air', 'completion_mode': 'gates',
        'wind_range': (0.0, 3.0), 'turb_range': (1.0, 3.0),
        'dr_scale': 0.8,
        'thermals': True, 'projectiles': False, 'ew': False,
        'rewards': {
            'survival_reward': 0.0,
            'altitude_coeff': 0.005,
            'stability_coeff': 0.015,
            'orientation_coeff': 0.015,
            'smoothness_coeff': 0.01,
            'drift_coeff': 0.0,
            'energy_coeff': 0.001,
            'speed_coeff': 0.002,
            'waypoint_reward': 5.0,
            'progress_reward': 0.05,
            'course_complete_reward': 50.0,
            'gate_align_bonus': 1.0,
            'dodge_bonus': 0.0,
            'timeout_penalty': TIMEOUT_PENALTY,
            'collision_penalty': -10.0,
            'crash_penalty': -10.0,
            'oob_penalty': -10.0,
            'projectile_hit_penalty': -5.0,
        },
    },
    # ══ Phase D: Dynamic Threats ═══════════════════════════════
    14: {
        'name': 'Combat Course',
        'max_steps': 2500,  # 50s — 7 gates + moving walls + projectile dodging
        'start_mode': 'air', 'completion_mode': 'gates',
        'wind_range': (0.0, 4.0), 'turb_range': (1.0, 3.5),
        'dr_scale': 1.0,
        'thermals': True, 'projectiles': True, 'ew': False,
        'turret_lead': 0.1,  # gentle lead targeting — first exposure to projectiles
        'rewards': {
            'survival_reward': 0.0,
            'altitude_coeff': 0.005,
            'stability_coeff': 0.01,
            'orientation_coeff': 0.01,
            'smoothness_coeff': 0.005,
            'drift_coeff': 0.0,
            'energy_coeff': 0.001,
            'speed_coeff': 0.001,
            'waypoint_reward': 8.0,
            'progress_reward': 0.10,
            'course_complete_reward': 50.0,
            'gate_align_bonus': 1.0,
            'dodge_bonus': 0.5,
            'proj_avoid_bonus': 0.01,
            'timeout_penalty': TIMEOUT_PENALTY,
            'collision_penalty': -10.0,
            'crash_penalty': -10.0,
            'oob_penalty': -10.0,
            'projectile_hit_penalty': -2.0,
        },
    },
    # ══ Phase E: Contested Environment ═════════════════════════
    15: {
        'name': 'Final Course',
        'max_steps': 3000,  # 60s — 7 gates + all threats + EW + heavy wind
        'start_mode': 'air', 'completion_mode': 'gates',
        'wind_range': (0.0, 5.0), 'turb_range': (2.0, 4.5),
        'dr_scale': 1.0,
        'thermals': True, 'projectiles': True, 'ew': True,
        'turret_lead': 0.3,  # full lead targeting
        'rewards': {
            'survival_reward': 0.0,
            'altitude_coeff': 0.005,
            'stability_coeff': 0.01,
            'orientation_coeff': 0.01,
            'smoothness_coeff': 0.005,
            'drift_coeff': 0.0,
            'energy_coeff': 0.001,
            'speed_coeff': 0.002,
            'waypoint_reward': 6.0,
            'progress_reward': 0.08,
            'course_complete_reward': 50.0,
            'gate_align_bonus': 1.0,
            'dodge_bonus': 0.5,
            'proj_avoid_bonus': 0.005,
            'timeout_penalty': TIMEOUT_PENALTY,
            'collision_penalty': -10.0,
            'crash_penalty': -10.0,
            'oob_penalty': -10.0,
            'projectile_hit_penalty': -5.0,
        },
    },
}

MAX_CURRICULUM_LEVEL = 15

# ── Obstacle Types ────────────────────────────────────────────
OBS_WALL    = 0   # AABB box
OBS_COLUMN  = 1   # cylinder
OBS_GATE    = 2   # rectangular opening (waypoint)
OBS_MOVING  = 3   # moving AABB

# ── LIDAR Ray Directions (body frame, pre-computed) ───────────
# 12 rays: 4 cardinal, 4 diagonal, up, down, forward-up, forward-down
_c45 = np.cos(np.pi / 4)
_s30 = np.sin(np.pi / 6)
_c30 = np.cos(np.pi / 6)
LIDAR_DIRS = np.array([
    [ 1,  0,  0],    # forward
    [-1,  0,  0],    # backward
    [ 0,  1,  0],    # right
    [ 0, -1,  0],    # left
    [ _c45,  _c45, 0],  # forward-right
    [-_c45,  _c45, 0],  # backward-right
    [-_c45, -_c45, 0],  # backward-left
    [ _c45, -_c45, 0],  # forward-left
    [ 0,  0,  1],    # up
    [ 0,  0, -1],    # down
    [ _c30, 0, _s30],  # forward-up 30deg
    [ _c30, 0, -_s30], # forward-down 30deg
], dtype=np.float32)

# ── Projectile Constants ──────────────────────────────────────
MAX_PROJECTILES     = 5
PROJ_SPEED          = 15.0      # m/s
PROJ_RADIUS         = 0.1       # m
PROJ_LIFETIME       = 5.0       # s
PROJ_SPAWN_INTERVAL = 2.0       # s
MAX_TURRETS         = 6         # max gun installations per course
TURRET_HEIGHT       = 1.5       # pedestal height (m)

# ── Battery ───────────────────────────────────────────────────
BATTERY_CAPACITY    = 300.0     # seconds at hover


# ══════════════════════════════════════════════════════════════
#  Course Generation
# ══════════════════════════════════════════════════════════════

def _generate_course(level: int, rng: np.random.Generator) -> dict:
    """Generate obstacle course for a given difficulty level (1-15).

    Levels 1-9 are flight skills (hover/takeoff/land/altitude/yaw/fly-to — no gates).
    Level 9 (fly-to-point) generates a target position instead of gates.
    Levels 10+ progressively add gates, obstacles, moving walls.

    Returns dict with 'obstacles', 'gates', and optionally 'target_pos'.
    """
    obstacles = []
    gates = []
    mid_y = COURSE_D / 2.0

    # ── Levels 1-8: Empty (flight basics + intermediate skills) ──
    if level <= 8:
        return {'obstacles': obstacles, 'gates': gates}

    # ── Level 9: Fly to point — single target position ───────
    if level == 9:
        # Generate a random target 10-25m away from start
        tx = rng.uniform(12.0, 28.0)
        ty = rng.uniform(10.0, COURSE_D - 10.0)
        tz = rng.uniform(3.0, 8.0)
        return {
            'obstacles': obstacles,
            'gates': gates,
            'target_pos': np.array([tx, ty, tz], dtype=np.float32),
        }

    # ── Level 10-11: Wide gates in a straight line, no obstacles ──
    if level >= 10:
        for i in range(3):
            gx = 8.0 + i * 10.0
            gates.append({
                'center': np.array([gx, mid_y, 5.0], dtype=np.float32),
                'normal': np.array([1.0, 0.0, 0.0], dtype=np.float32),
                'up':     np.array([0.0, 0.0, 1.0], dtype=np.float32),
                'right':  np.array([0.0, 1.0, 0.0], dtype=np.float32),
                'width':  6.0,
                'height': 5.0,
                'sector': 10,
            })

    # ── Level 12-13: Add columns and walls between gates ─────
    if level >= 12:
        # 2 more gates (total 5)
        gates.append({
            'center': np.array([15.0, mid_y - 5.0, 4.0], dtype=np.float32),
            'normal': np.array([0.0, -1.0, 0.0], dtype=np.float32),
            'up':     np.array([0.0, 0.0, 1.0], dtype=np.float32),
            'right':  np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'width':  5.0,
            'height': 4.0,
            'sector': 12,
        })
        gates.append({
            'center': np.array([32.0, mid_y, 6.0], dtype=np.float32),
            'normal': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'up':     np.array([0.0, 0.0, 1.0], dtype=np.float32),
            'right':  np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'width':  5.0,
            'height': 4.5,
            'sector': 12,
        })
        # Columns
        for _ in range(4):
            cx = rng.uniform(5.0, 30.0)
            cy = rng.uniform(8.0, COURSE_D - 8.0)
            obstacles.append({
                'type': OBS_COLUMN,
                'x': cx, 'y': cy,
                'radius': rng.uniform(0.5, 1.5),
                'z_min': 0.0, 'z_max': rng.uniform(8.0, 15.0),
                'sector': 12,
            })
        # Wall segments
        obstacles.append({
            'type': OBS_WALL,
            'min': np.array([12.0, 10.0, 0.0], dtype=np.float32),
            'max': np.array([12.5, 16.0, 7.0], dtype=np.float32),
            'sector': 12,
        })
        obstacles.append({
            'type': OBS_WALL,
            'min': np.array([24.0, 24.0, 0.0], dtype=np.float32),
            'max': np.array([24.5, 30.0, 8.0], dtype=np.float32),
            'sector': 12,
        })

    # ── Level 14+: Moving walls and more gates ──────────────
    if level >= 14:
        # 2 more gates (total 7)
        gates.append({
            'center': np.array([22.0, mid_y + 6.0, 5.0], dtype=np.float32),
            'normal': np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'up':     np.array([0.0, 0.0, 1.0], dtype=np.float32),
            'right':  np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'width':  4.0,
            'height': 4.0,
            'sector': 14,
        })
        gates.append({
            'center': np.array([35.0, mid_y, 4.0], dtype=np.float32),
            'normal': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'up':     np.array([0.0, 0.0, 1.0], dtype=np.float32),
            'right':  np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'width':  3.5,
            'height': 3.5,
            'sector': 14,
        })
        # Moving walls (sinusoidal)
        for i in range(2):
            my = 10.0 + i * 15.0
            obstacles.append({
                'type': OBS_MOVING,
                'base_min': np.array([26.0, my, 0.0], dtype=np.float32),
                'base_max': np.array([27.0, my + 3.0, 10.0], dtype=np.float32),
                'min': np.array([26.0, my, 0.0], dtype=np.float32),
                'max': np.array([27.0, my + 3.0, 10.0], dtype=np.float32),
                'axis': 1,
                'amplitude': 5.0,
                'period': 4.0,
                'sector': 14,
            })

    # ── Level 15: Narrower gates, more obstacles ─────────────
    if level >= 15:
        gates.append({
            'center': np.array([37.0, mid_y, 7.0], dtype=np.float32),
            'normal': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'up':     np.array([0.0, 0.0, 1.0], dtype=np.float32),
            'right':  np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'width':  3.0,
            'height': 3.0,
            'sector': 15,
        })
        # Extra columns in EW zone
        for _ in range(3):
            cx = rng.uniform(30.0, 38.0)
            cy = rng.uniform(10.0, COURSE_D - 10.0)
            obstacles.append({
                'type': OBS_COLUMN,
                'x': cx, 'y': cy,
                'radius': rng.uniform(0.3, 1.0),
                'z_min': 0.0, 'z_max': rng.uniform(6.0, 12.0),
                'sector': 15,
            })
        # Tight final gate
        gates.append({
            'center': np.array([39.0, mid_y, 5.0], dtype=np.float32),
            'normal': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'up':     np.array([0.0, 0.0, 1.0], dtype=np.float32),
            'right':  np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'width':  2.5,
            'height': 2.5,
            'sector': 15,
        })
        # Extra moving obstacle
        obstacles.append({
            'type': OBS_MOVING,
            'base_min': np.array([34.0, 16.0, 0.0], dtype=np.float32),
            'base_max': np.array([35.0, 19.0, 12.0], dtype=np.float32),
            'min': np.array([34.0, 16.0, 0.0], dtype=np.float32),
            'max': np.array([35.0, 19.0, 12.0], dtype=np.float32),
            'axis': 0,
            'amplitude': 3.0,
            'period': 3.0,
            'sector': 15,
        })

    # ── Gun turrets (placed at arena edges for levels with projectiles) ──
    turrets = _place_turrets(level, gates, rng)

    return {'obstacles': obstacles, 'gates': gates, 'turrets': turrets}


def _place_turrets(level: int, gates: list, rng, n_turrets: int = None,
                   difficulty: float = None) -> list:
    """Place gun turret installations along arena edges.

    Turrets are positioned on the arena perimeter at ground level,
    spread out to provide cross-fire coverage of the course.
    For fixed curriculum: 2-3 turrets at levels 13+.
    For endless mode: count scales with difficulty.
    """
    # Determine turret count
    if n_turrets is not None:
        count = n_turrets
    elif level >= 15:
        count = 3
    elif level >= 13:
        count = 2
    else:
        return []

    count = min(count, MAX_TURRETS)
    if not gates:
        return []

    # Compute course centroid for aim reference
    cx = np.mean([g['center'][0] for g in gates])
    cy = np.mean([g['center'][1] for g in gates])

    # Candidate positions along arena edges (ground level + pedestal)
    z = TURRET_HEIGHT
    candidates = []
    # Bottom edge (y=0)
    for frac in [0.2, 0.4, 0.6, 0.8]:
        candidates.append(np.array([frac * COURSE_W, 0.5, z], dtype=np.float32))
    # Top edge (y=COURSE_D)
    for frac in [0.2, 0.4, 0.6, 0.8]:
        candidates.append(np.array([frac * COURSE_W, COURSE_D - 0.5, z], dtype=np.float32))
    # Left edge (x=0)
    for frac in [0.3, 0.5, 0.7]:
        candidates.append(np.array([0.5, frac * COURSE_D, z], dtype=np.float32))
    # Right edge (x=COURSE_W)
    for frac in [0.3, 0.5, 0.7]:
        candidates.append(np.array([COURSE_W - 0.5, frac * COURSE_D, z], dtype=np.float32))

    # Score candidates: prefer those with clear sight to course centroid
    # and spread turrets apart
    rng.shuffle(candidates)
    turrets = []
    for _ in range(count):
        best_score = -1e9
        best_idx = 0
        for ci, cand in enumerate(candidates):
            # Distance to course center (prefer middle range — not too close, not too far)
            d = np.sqrt((cand[0] - cx)**2 + (cand[1] - cy)**2)
            dist_score = -abs(d - 15.0)  # prefer ~15m from center
            # Separation from already-placed turrets
            sep_score = 0.0
            for t in turrets:
                sep = np.sqrt((cand[0] - t['pos'][0])**2 + (cand[1] - t['pos'][1])**2)
                sep_score += min(sep, 20.0)
            # Slight randomness
            noise = rng.uniform(-2.0, 2.0)
            score = dist_score + sep_score * 0.5 + noise
            if score > best_score:
                best_score = score
                best_idx = ci
        chosen = candidates.pop(best_idx)
        turrets.append({
            'pos': chosen,
            'id': len(turrets),
        })
        if not candidates:
            break

    return turrets


# ══════════════════════════════════════════════════════════════
#  Endless Mode — Procedural Course Generation
# ══════════════════════════════════════════════════════════════

def _generate_endless_course(scenario: int, base_seed: int, difficulty: float) -> dict:
    """Generate a seed-deterministic procedural course for endless mode.

    Each scenario has a unique seed: base_seed * 100_000 + scenario.
    Difficulty D (1.0-5.0) parametrically scales all element counts/constraints.
    Gate placement uses a path-graph algorithm: chain of waypoints progressing
    left→right with obstacles placed between gate pairs.
    """
    seed = base_seed * 100_000 + scenario
    rng = np.random.default_rng(seed)
    D = difficulty

    # ── Element counts from difficulty ──────────────────────
    n_gates       = 3 + int(D * 1.5)
    n_columns     = 2 + int(D * 1.2)
    n_walls       = 1 + int(D * 0.8)
    n_moving      = int(D * 0.7)
    n_thermals    = 1 + int(D * 0.5)
    n_ew          = int(D * 0.4)
    gate_w        = max(2.0, 6.0 - D * 0.7)
    gate_h        = max(2.0, 5.0 - D * 0.5)

    margin = 2.0  # arena edge margin

    # ── Path-graph gate placement ───────────────────────────
    # Chain of waypoints progressing left→right (x=5 to x=38)
    gates = []
    prev_pt = np.array([2.0, COURSE_D / 2.0, 3.0], dtype=np.float32)
    gate_centers = [prev_pt.copy()]  # include start for obstacle placement

    x_cursor = 5.0
    for gi in range(n_gates):
        # Progress x by 5-12m, clamped to arena
        dx = rng.uniform(5.0, 12.0)
        x_cursor = min(x_cursor + dx, COURSE_W - margin) if gi > 0 else x_cursor
        # Y/Z jitter scales with difficulty
        jitter_y = rng.uniform(-3.0 * D, 3.0 * D)
        jitter_z = rng.uniform(-1.5 * D, 1.5 * D)
        cy = np.clip(COURSE_D / 2.0 + jitter_y, margin + 2.0, COURSE_D - margin - 2.0)
        cz = np.clip(3.0 + jitter_z, 2.0, COURSE_H - 3.0)
        center = np.array([x_cursor, cy, cz], dtype=np.float32)

        # Gate faces approach direction with angular offset
        approach = center - prev_pt
        approach_len = np.linalg.norm(approach)
        if approach_len > 0.1:
            normal = approach / approach_len
        else:
            normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Random angular offset ±15°×D
        angle_offset = rng.uniform(-0.26 * D, 0.26 * D)  # 0.26 rad ≈ 15°
        cos_a, sin_a = np.cos(angle_offset), np.sin(angle_offset)
        normal_rot = np.array([
            normal[0] * cos_a - normal[1] * sin_a,
            normal[0] * sin_a + normal[1] * cos_a,
            normal[2],
        ], dtype=np.float32)
        normal_rot /= np.linalg.norm(normal_rot) + 1e-8

        # Up is always world-Z, right = cross(normal, up)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(normal_rot, up)
        right_len = np.linalg.norm(right)
        if right_len > 0.01:
            right /= right_len
        else:
            right = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        gates.append({
            'center': center,
            'normal': normal_rot,
            'up': up,
            'right': right,
            'width': gate_w,
            'height': gate_h,
            'sector': 16 + gi,
        })
        gate_centers.append(center.copy())
        prev_pt = center
        x_cursor = center[0]

    # ── Obstacle placement (between gate pairs) ─────────────
    obstacles = []
    min_gate_dist = 2.0
    min_path_dist = DRONE_RADIUS * 3.0

    def _far_from_gates(px, py, pz, radius=0.0):
        """Check point is at least min_gate_dist from all gate centers."""
        for gc in gate_centers:
            d = np.sqrt((px - gc[0])**2 + (py - gc[1])**2 + (pz - gc[2])**2)
            if d < min_gate_dist + radius:
                return False
        return True

    def _far_from_path(px, py, pz):
        """Check point is at least min_path_dist from direct gate-to-gate lines."""
        for k in range(len(gate_centers) - 1):
            a = gate_centers[k]
            b = gate_centers[k + 1]
            ab = b - a
            ab_len = np.linalg.norm(ab)
            if ab_len < 0.1:
                continue
            ab_n = ab / ab_len
            pt = np.array([px, py, pz], dtype=np.float32)
            ap = pt - a
            t = np.clip(np.dot(ap, ab_n), 0.0, ab_len)
            closest = a + ab_n * t
            dist = np.linalg.norm(pt - closest)
            if dist < min_path_dist:
                return False
        return True

    # Static columns
    for _ in range(n_columns):
        for _try in range(20):
            cx = rng.uniform(margin + 3.0, COURSE_W - margin - 1.0)
            cy = rng.uniform(margin + 3.0, COURSE_D - margin - 1.0)
            cr = rng.uniform(0.3, 1.2)
            cz_max = rng.uniform(6.0, COURSE_H - 2.0)
            if _far_from_gates(cx, cy, cz_max / 2.0, cr) and _far_from_path(cx, cy, cz_max / 2.0):
                obstacles.append({
                    'type': OBS_COLUMN,
                    'x': cx, 'y': cy,
                    'radius': cr,
                    'z_min': 0.0, 'z_max': cz_max,
                    'sector': 16,
                })
                break

    # Static walls
    for _ in range(n_walls):
        for _try in range(20):
            wx = rng.uniform(margin + 3.0, COURSE_W - margin - 3.0)
            wy = rng.uniform(margin + 3.0, COURSE_D - margin - 3.0)
            wz_max = rng.uniform(5.0, COURSE_H - 3.0)
            w_len = rng.uniform(3.0, 6.0)
            w_thick = 0.5
            # Random orientation: along X or Y
            if rng.random() > 0.5:
                mn = np.array([wx, wy, 0.0], dtype=np.float32)
                mx = np.array([wx + w_thick, wy + w_len, wz_max], dtype=np.float32)
            else:
                mn = np.array([wx, wy, 0.0], dtype=np.float32)
                mx = np.array([wx + w_len, wy + w_thick, wz_max], dtype=np.float32)
            mid = (mn + mx) / 2.0
            if _far_from_gates(mid[0], mid[1], mid[2]) and _far_from_path(mid[0], mid[1], mid[2]):
                obstacles.append({
                    'type': OBS_WALL,
                    'min': mn, 'max': mx,
                    'sector': 16,
                })
                break

    # Moving walls
    for _ in range(n_moving):
        for _try in range(20):
            mx_pos = rng.uniform(margin + 5.0, COURSE_W - margin - 5.0)
            my_pos = rng.uniform(margin + 5.0, COURSE_D - margin - 5.0)
            mz_max = rng.uniform(6.0, COURSE_H - 2.0)
            m_len = rng.uniform(2.0, 4.0)
            axis = int(rng.integers(0, 2))  # 0=X, 1=Y
            amp = rng.uniform(2.0, 5.0)
            period = rng.uniform(3.0, 6.0)
            mn = np.array([mx_pos, my_pos, 0.0], dtype=np.float32)
            mx_arr = np.array([mx_pos + 1.0, my_pos + m_len, mz_max], dtype=np.float32)
            mid = (mn + mx_arr) / 2.0
            if _far_from_gates(mid[0], mid[1], mid[2]) and _far_from_path(mid[0], mid[1], mid[2]):
                obstacles.append({
                    'type': OBS_MOVING,
                    'base_min': mn.copy(), 'base_max': mx_arr.copy(),
                    'min': mn.copy(), 'max': mx_arr.copy(),
                    'axis': axis,
                    'amplitude': amp,
                    'period': period,
                    'sector': 16,
                })
                break

    # ── Thermals and EW zones (outside gate corridors) ──────
    thermals = []
    for _ in range(n_thermals):
        for _try in range(15):
            tx = rng.uniform(margin + 5.0, COURSE_W - margin - 5.0)
            ty = rng.uniform(margin + 5.0, COURSE_D - margin - 5.0)
            tr = rng.uniform(2.0, 4.0)
            ts = rng.uniform(5.0, 12.0)
            if _far_from_gates(tx, ty, 5.0, tr):
                thermals.append({'x': tx, 'y': ty, 'radius': tr, 'strength': ts})
                break

    gps_denial = []
    jamming = []
    for _ in range(n_ew):
        for _try in range(15):
            ex = rng.uniform(margin + 8.0, COURSE_W - margin - 5.0)
            ey = rng.uniform(margin + 8.0, COURSE_D - margin - 5.0)
            er = rng.uniform(5.0, 8.0)
            if _far_from_gates(ex, ey, 5.0, er):
                gps_denial.append({
                    'center': np.array([ex, ey], dtype=np.float32),
                    'radius': er,
                })
                jamming.append({
                    'center': np.array([ex + rng.uniform(-2, 2), ey + rng.uniform(-2, 2)],
                                       dtype=np.float32),
                    'radius': er * 0.8,
                    'intensity': min(0.9, 0.4 + D * 0.1),
                })
                break

    # ── Gun turrets (scale with difficulty) ─────────────────
    n_turrets = 1 + int(D * 0.5)
    turrets = _place_turrets(0, gates, rng, n_turrets=n_turrets, difficulty=D)

    return {
        'obstacles': obstacles,
        'gates': gates,
        'thermals': thermals,
        'gps_denial': gps_denial,
        'jamming': jamming,
        'turrets': turrets,
    }


def _validate_endless_course(course: dict) -> bool:
    """Validate that an endless course is flyable.

    Checks:
    - Ray-cast between consecutive gate centers doesn't intersect obstacles
    - Consecutive gates 4-20m apart
    - Gate centers have 1.5m clearance from obstacles
    - All positions within arena bounds
    """
    gates = course['gates']
    obstacles = course['obstacles']
    if len(gates) < 2:
        return False

    for i in range(len(gates)):
        gc = gates[i]['center']
        # Check arena bounds with margin
        if gc[0] < 1.0 or gc[0] > COURSE_W - 1.0:
            return False
        if gc[1] < 1.0 or gc[1] > COURSE_D - 1.0:
            return False
        if gc[2] < 1.0 or gc[2] > COURSE_H - 1.0:
            return False

    for i in range(len(gates) - 1):
        a = gates[i]['center']
        b = gates[i + 1]['center']
        dist = np.linalg.norm(b - a)
        if dist < 4.0 or dist > 25.0:
            return False

        # Simple obstacle clearance check: sample points along gate-to-gate line
        n_samples = max(5, int(dist / 0.5))
        for s in range(n_samples + 1):
            t = s / n_samples
            pt = a + (b - a) * t
            for obs in obstacles:
                if obs['type'] == OBS_COLUMN:
                    dx = pt[0] - obs['x']
                    dy = pt[1] - obs['y']
                    h_dist = np.sqrt(dx * dx + dy * dy)
                    if h_dist < obs['radius'] + 1.5 and obs['z_min'] <= pt[2] <= obs['z_max']:
                        return False
                elif obs['type'] in (OBS_WALL, OBS_MOVING):
                    mn = obs['min'] if obs['type'] == OBS_WALL else obs['base_min']
                    mx = obs['max'] if obs['type'] == OBS_WALL else obs['base_max']
                    # Expand by 1.5m for clearance
                    if (mn[0] - 1.5 <= pt[0] <= mx[0] + 1.5 and
                        mn[1] - 1.5 <= pt[1] <= mx[1] + 1.5 and
                        mn[2] - 1.5 <= pt[2] <= mx[2] + 1.5):
                        return False

    return True


# ══════════════════════════════════════════════════════════════
#  Vectorized Drone Environment
# ══════════════════════════════════════════════════════════════

class VectorizedDroneEnv:
    """Vectorized quadrotor environment with real-physics rigid body dynamics.

    Runs N parallel environments. Each environment has one drone navigating
    an obstacle course with wind, projectiles, and electronic warfare.
    """

    def __init__(self, n_envs: int = 64, max_steps: int = 1000):
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.rng = np.random.default_rng(42)

        # ── Drone state arrays [N, ...] ───────────────────────
        self.pos = np.zeros((n_envs, 3), dtype=np.float32)       # world position
        self.vel = np.zeros((n_envs, 3), dtype=np.float32)       # world velocity
        self.quat = np.zeros((n_envs, 4), dtype=np.float32)      # orientation [w,x,y,z]
        self.ang_vel = np.zeros((n_envs, 3), dtype=np.float32)   # body-frame angular vel
        self.motor_omega = np.zeros((n_envs, 4), dtype=np.float32)  # current motor rad/s
        self.motor_cmd = np.zeros((n_envs, 4), dtype=np.float32)   # commanded motor rad/s
        self.prev_action = np.zeros((n_envs, 4), dtype=np.float32)  # last raw action [-1,1]

        # IMU readings (body frame, updated each substep)
        self.imu_accel = np.zeros((n_envs, 3), dtype=np.float32)
        self.imu_gyro = np.zeros((n_envs, 3), dtype=np.float32)

        # ── Wind state ────────────────────────────────────────
        self.base_wind = np.zeros((n_envs, 3), dtype=np.float32)
        self.episode_wind = np.zeros((n_envs, 3), dtype=np.float32)
        self.turb_state = np.zeros((n_envs, 3), dtype=np.float32)
        self.turb_intensity = np.full(n_envs, 1.5, dtype=np.float32)
        self.wind_force = np.zeros((n_envs, 3), dtype=np.float32)

        # Wind gusts
        self.gust_dir = np.zeros((n_envs, 3), dtype=np.float32)
        self.gust_strength = np.zeros(n_envs, dtype=np.float32)
        self.gust_remaining = np.zeros(n_envs, dtype=np.float32)

        # Thermal zones (set per course layout)
        self.thermal_zones = []

        # ── Navigation state ──────────────────────────────────
        self.current_gate = np.zeros(n_envs, dtype=np.int32)
        self.prev_waypoint_dist = np.zeros(n_envs, dtype=np.float32)
        self.step_count = np.zeros(n_envs, dtype=np.int32)
        self.sim_time = np.zeros(n_envs, dtype=np.float32)

        # ── Projectile state ──────────────────────────────────
        self.proj_pos = np.zeros((n_envs, MAX_PROJECTILES, 3), dtype=np.float32)
        self.proj_vel = np.zeros((n_envs, MAX_PROJECTILES, 3), dtype=np.float32)
        self.proj_active = np.zeros((n_envs, MAX_PROJECTILES), dtype=bool)
        self.proj_life = np.zeros((n_envs, MAX_PROJECTILES), dtype=np.float32)
        self.proj_spawn_timer = np.zeros(n_envs, dtype=np.float32)
        self.proj_source_turret = np.full((n_envs, MAX_PROJECTILES), -1, dtype=np.int32)
        self._turret_flash = {}  # turret_id → remaining flash time (for env 0)

        # ── EW state ──────────────────────────────────────────
        self.gps_denied = np.zeros(n_envs, dtype=bool)
        self.jamming_intensity = np.zeros(n_envs, dtype=np.float32)
        self.gps_denial_zones = []
        self.jamming_zones = []

        # ── Battery state ─────────────────────────────────────
        self.battery = np.ones(n_envs, dtype=np.float32)

        # ── Hit tracking ──────────────────────────────────────
        self.hits_taken = np.zeros(n_envs, dtype=np.int32)
        self.max_hits = 3

        # ── Domain randomization parameters (per-env) ─────────
        self.dr_mass = np.full(n_envs, MASS, dtype=np.float32)
        self.dr_ixx = np.full(n_envs, IXX, dtype=np.float32)
        self.dr_iyy = np.full(n_envs, IYY, dtype=np.float32)
        self.dr_izz = np.full(n_envs, IZZ, dtype=np.float32)
        self.dr_kf = np.full(n_envs, KF, dtype=np.float32)
        self.dr_km = np.full(n_envs, KM, dtype=np.float32)
        self.dr_motor_tau = np.full(n_envs, MOTOR_TAU, dtype=np.float32)
        self.dr_drag_xy = np.full(n_envs, DRAG_XY, dtype=np.float32)

        # ── Course ────────────────────────────────────────────
        self.curriculum_level = 1
        self.course_data = _generate_course(self.curriculum_level, self.rng)
        self.obstacles = self.course_data['obstacles']
        self.gates = self.course_data['gates']
        self.total_gates = len(self.gates)
        self.turrets = self.course_data.get('turrets', [])
        self._turret_fire_idx = 0  # round-robin turret selection

        # ── Curriculum tracking ───────────────────────────────
        self.recent_completions = []
        self.curriculum_window = 100

        # ── Endless mode state ───────────────────────────────
        self._endless_mode = False
        self._endless_scenario = 0
        self._endless_difficulty = 1.0
        self._endless_base_seed = 42
        self._endless_best_streak = 0
        self._endless_current_streak = 0
        self._course_changed = False  # flag for broadcast loop

        # ── User weapon state ────────────────────────────────
        self._weapon_mode = False
        self._user_shots = 0
        self._user_hits = 0

        # ── Adversary drone state (env 0 only) ──────────────
        self._adversary_enabled = False
        self._adversary_lethal = False
        self.adv_pos = np.zeros(3, dtype=np.float32)
        self.adv_vel = np.zeros(3, dtype=np.float32)
        self.adv_active = False
        self.adv_lock_lost_timer = 0.0

        # ── Hover / takeoff / landing state ──────────────────
        self.hover_duration = np.zeros(n_envs, dtype=np.float32)
        self.takeoff_hold_time = np.zeros(n_envs, dtype=np.float32)
        self.start_pos = np.zeros((n_envs, 3), dtype=np.float32)
        self.hover_shrink_stage = np.zeros(n_envs, dtype=np.int32)  # 0..HOVER_SHRINK_STAGES

        # ── Ground thrash tracking ─────────────────────────
        self.ground_time = np.zeros(n_envs, dtype=np.float32)  # seconds on ground

        # ── Altitude change / yaw / fly-to-point state ──────
        self.alt_cylinders = np.zeros((n_envs, 3, 3), dtype=np.float32)  # 3 cylinder centers [x,y,z]
        self.alt_current_cyl = np.zeros(n_envs, dtype=np.int32)  # which cylinder is active (0-2)
        self.alt_changes_done = np.zeros(n_envs, dtype=np.int32)
        self.alt_hold_time = np.zeros(n_envs, dtype=np.float32)
        self.target_yaw = np.zeros(n_envs, dtype=np.float32)
        self.yaw_changes_done = np.zeros(n_envs, dtype=np.int32)
        self.yaw_hold_time = np.zeros(n_envs, dtype=np.float32)
        self.fly_to_target = np.zeros((n_envs, 3), dtype=np.float32)
        self.fly_to_hold_time = np.zeros(n_envs, dtype=np.float32)

        # ── Per-level environment config (set by _apply_curriculum_profile) ──
        self._wind_range = (0.0, 0.0)
        self._turb_range = (0.0, 0.0)
        self._dr_scale = 0.3
        self._start_mode = 'air'
        self._completion_mode = 'hover'
        self._hover_stages = HOVER_SHRINK_STAGES

        # ── Configurable reward weights (set by _apply_curriculum_profile) ──
        self._turret_lead_factor = 0.3  # default lead targeting
        self.waypoint_reward = WAYPOINT_REWARD
        self.progress_reward = PROGRESS_REWARD
        self.course_complete_reward = COURSE_COMPLETE_REWARD
        self.survival_reward = SURVIVAL_REWARD
        self.collision_penalty = COLLISION_PENALTY
        self.crash_penalty = CRASH_PENALTY
        self.oob_penalty = OOB_PENALTY
        self.projectile_hit_penalty = PROJECTILE_HIT_PENALTY
        self.dodge_bonus = DODGE_BONUS
        self.proj_avoid_bonus = 0.0
        self.stability_coeff = STABILITY_COEFF
        self.energy_coeff = ENERGY_COEFF
        self.smoothness_coeff = SMOOTHNESS_COEFF
        self.altitude_coeff = ALTITUDE_COEFF
        self.orientation_coeff = ORIENTATION_COEFF
        self.speed_coeff = SPEED_COEFF
        self.gate_align_bonus = GATE_ALIGN_BONUS
        self.drift_coeff = DRIFT_COEFF
        self.timeout_penalty = 0.0  # only applied on gate courses

        # Previous motor command (for smoothness penalty)
        self.prev_motor_cmd = np.full((n_envs, 4), HOVER_OMEGA, dtype=np.float32)

        # ── LIDAR cache ───────────────────────────────────────
        self.lidar_distances = np.full((n_envs, 12), MAX_LIDAR_RANGE, dtype=np.float32)

        # Apply level-1 curriculum profile (sets rewards, wind, DR)
        self._apply_curriculum_profile()

        # Initialize
        self.reset()

    # ── Properties for PPO compatibility ──────────────────────
    @property
    def n_active(self):
        return 1  # single drone per env

    @property
    def players_per_team(self):
        return 1  # compatibility shim

    # ══════════════════════════════════════════════════════════
    #  Quaternion Helpers (vectorized)
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _quat_rotate(q, v):
        """Rotate vectors v by quaternions q. q=[w,x,y,z]. [N,4] x [N,3] -> [N,3]"""
        qvec = q[:, 1:4]                         # [N, 3]
        uv = np.cross(qvec, v)                    # [N, 3]
        uuv = np.cross(qvec, uv)                  # [N, 3]
        return v + 2.0 * (q[:, 0:1] * uv + uuv)  # [N, 3]

    @staticmethod
    def _quat_rotate_inv(q, v):
        """Inverse rotation (conjugate q then rotate). [N,4] x [N,3] -> [N,3]"""
        q_conj = q.copy()
        q_conj[:, 1:4] *= -1
        qvec = q_conj[:, 1:4]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2.0 * (q_conj[:, 0:1] * uv + uuv)

    @staticmethod
    def _quat_to_euler(q):
        """Convert quaternion to Euler angles (roll, pitch, yaw). [N,4] -> [N,3]"""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.stack([roll, pitch, yaw], axis=-1)

    # ══════════════════════════════════════════════════════════
    #  Curriculum Profile Application
    # ══════════════════════════════════════════════════════════

    def _apply_curriculum_profile(self):
        """Apply the current curriculum level's reward weights, wind, and DR settings.

        For endless mode (level > 15), builds a dynamic profile from difficulty D
        instead of looking up a static CURRICULUM_PROFILES entry.
        """
        if self._endless_mode:
            self._apply_endless_profile()
            return

        profile = CURRICULUM_PROFILES[self.curriculum_level]
        self.max_steps = profile['max_steps']
        self._wind_range = profile['wind_range']
        self._turb_range = profile['turb_range']
        self._dr_scale = profile['dr_scale']
        self._start_mode = profile.get('start_mode', 'air')
        self._completion_mode = profile.get('completion_mode', 'gates')
        self._hover_stages = profile.get('hover_stages', HOVER_SHRINK_STAGES)

        # Apply reward weights from profile
        for key, val in profile['rewards'].items():
            setattr(self, key, val)

        # Turret lead targeting factor (how accurately turrets lead the drone)
        self._turret_lead_factor = profile.get('turret_lead', 0.3)

        # Feature flags: thermals, projectiles, EW zones
        if profile.get('thermals'):
            self.thermal_zones = [
                {'x': 30.0, 'y': 25.0, 'radius': 3.0, 'strength': 8.0},
                {'x': 35.0, 'y': 15.0, 'radius': 4.0, 'strength': 12.0},
            ]
        else:
            self.thermal_zones = []

        if profile.get('ew'):
            self.gps_denial_zones = [
                {'center': np.array([35.0, 20.0], dtype=np.float32), 'radius': 8.0},
            ]
            self.jamming_zones = [
                {'center': np.array([36.0, 15.0], dtype=np.float32),
                 'radius': 6.0, 'intensity': 0.8},
            ]
        else:
            self.gps_denial_zones = []
            self.jamming_zones = []

    def _apply_endless_profile(self):
        """Build a dynamic curriculum profile from endless difficulty D."""
        D = self._endless_difficulty
        self.max_steps = int(min(4000, 2500 + D * 300))
        self._wind_range = (0.0, min(6.0, D * 1.5))
        self._turb_range = (0.5 * D, min(5.0, D * 1.2))
        self._dr_scale = min(1.0, 0.7 + D * 0.06)
        self._start_mode = 'air'
        self._completion_mode = 'gates'

        # Reward weights — same as level 15 but with tighter projectile intervals
        self.survival_reward = 0.0
        self.timeout_penalty = TIMEOUT_PENALTY
        self.altitude_coeff = 0.005
        self.stability_coeff = 0.01
        self.orientation_coeff = 0.01
        self.smoothness_coeff = 0.005
        self.drift_coeff = 0.0
        self.energy_coeff = 0.001
        self.speed_coeff = 0.002
        self.waypoint_reward = 5.0
        self.progress_reward = 0.05
        self.course_complete_reward = 50.0
        self.gate_align_bonus = 1.0
        self.dodge_bonus = 0.5
        self.proj_avoid_bonus = 0.005
        self.collision_penalty = -10.0
        self.crash_penalty = -10.0
        self.oob_penalty = -10.0
        self.projectile_hit_penalty = -5.0
        self._turret_lead_factor = min(0.3, 0.1 + D * 0.05)  # ramps with difficulty

        # Thermals from endless course data
        if 'thermals' in self.course_data:
            self.thermal_zones = self.course_data['thermals']
        else:
            self.thermal_zones = []

        # EW from endless course data
        if 'gps_denial' in self.course_data:
            self.gps_denial_zones = self.course_data['gps_denial']
            self.jamming_zones = self.course_data.get('jamming', [])
        else:
            self.gps_denial_zones = []
            self.jamming_zones = []

    def set_endless_mode(self, enabled: bool, base_seed: int = None):
        """Toggle endless mode. When enabled, generates the first scenario."""
        self._endless_mode = enabled
        if base_seed is not None:
            self._endless_base_seed = base_seed
        if enabled:
            self._endless_scenario = 1
            self._endless_difficulty = 1.0
            self._endless_current_streak = 0
            self.curriculum_level = 16  # signal "beyond curriculum"
            self._generate_and_apply_endless()
        else:
            self._endless_scenario = 0
            self._endless_difficulty = 1.0
            self.curriculum_level = MAX_CURRICULUM_LEVEL
            self.course_data = _generate_course(self.curriculum_level, self.rng)
            self.obstacles = self.course_data['obstacles']
            self.gates = self.course_data['gates']
            self.total_gates = len(self.gates)
            self.turrets = self.course_data.get('turrets', [])
            self._apply_curriculum_profile()
            self.reset()

    def _generate_and_apply_endless(self):
        """Generate a validated endless course and apply its profile."""
        D = self._endless_difficulty
        # Try up to 5 seeds, fall back to gates-only
        for attempt in range(5):
            course = _generate_endless_course(
                self._endless_scenario + attempt * 1000,
                self._endless_base_seed, D,
            )
            if _validate_endless_course(course):
                break
        else:
            # Fallback: gates only (strip obstacles)
            course['obstacles'] = []

        self.course_data = course
        self.obstacles = course['obstacles']
        self.gates = course['gates']
        self.total_gates = len(self.gates)
        self.turrets = course.get('turrets', [])
        self._course_changed = True
        self._apply_curriculum_profile()
        self.reset()

    def _advance_endless_scenario(self):
        """Advance to next endless scenario after completion."""
        self._endless_scenario += 1
        self._endless_difficulty = min(5.0, 1.0 + 0.15 * self._endless_scenario)
        self._endless_current_streak += 1
        if self._endless_current_streak > self._endless_best_streak:
            self._endless_best_streak = self._endless_current_streak
        self._generate_and_apply_endless()

    # ══════════════════════════════════════════════════════════
    #  Reset
    # ══════════════════════════════════════════════════════════

    def reset(self, env_mask=None):
        """Reset environments. If env_mask is None, reset all."""
        if env_mask is None:
            env_mask = np.ones(self.n_envs, dtype=bool)

        n_reset = env_mask.sum()

        # Start position depends on curriculum start_mode
        if self._start_mode == 'ground':
            # Start on ground — drone must take off
            start_z = DRONE_RADIUS
            motor_start = 0.0  # motors off
        elif self._start_mode == 'hover':
            # Start hovering at 5m — drone must land
            start_z = 5.0
            motor_start = HOVER_OMEGA
        else:  # 'air' — default
            # Start at low altitude, already airborne
            start_z = 2.0
            motor_start = HOVER_OMEGA

        # Skill-training modes (hover, takeoff, land, altitude, yaw, fly-to-point)
        # start at arena center so targets don't clip outside bounds.
        # Gate modes start near the course entrance (x=2).
        if self._completion_mode in ('gates',):
            start_xy = np.array([2.0, COURSE_D / 2.0], dtype=np.float32)
        else:
            start_xy = np.array([COURSE_W / 2.0, COURSE_D / 2.0], dtype=np.float32)
        self.pos[env_mask] = np.array([start_xy[0], start_xy[1], start_z], dtype=np.float32)
        self.vel[env_mask] = 0.0
        # Identity quaternion: [w=1, x=0, y=0, z=0] = level attitude
        self.quat[env_mask] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.ang_vel[env_mask] = 0.0
        # Motors
        self.motor_omega[env_mask] = motor_start
        self.motor_cmd[env_mask] = motor_start
        self.prev_motor_cmd[env_mask] = motor_start
        self.prev_action[env_mask] = 0.0  # neutral action at episode start

        # Record start position for drift tracking
        self.start_pos[env_mask] = self.pos[env_mask].copy()

        self.imu_accel[env_mask] = np.array([0.0, 0.0, GRAVITY], dtype=np.float32)
        self.imu_gyro[env_mask] = 0.0

        # Navigation
        self.current_gate[env_mask] = 0
        self.step_count[env_mask] = 0
        self.sim_time[env_mask] = 0.0
        self.hits_taken[env_mask] = 0

        # Hover / takeoff / landing state
        self.hover_duration[env_mask] = 0.0
        self.takeoff_hold_time[env_mask] = 0.0
        self.hover_shrink_stage[env_mask] = 0

        # Altitude change state — 3 cylinders stacked vertically (same XY, different Z)
        # Drone must move up/down to enter each cylinder zone in random order
        self.alt_hold_time[env_mask] = 0.0
        self.alt_changes_done[env_mask] = 0
        self.alt_current_cyl[env_mask] = 0
        for idx in np.where(env_mask)[0]:
            # All 3 cylinders share the same XY (near drone start position)
            cx = self.start_pos[idx, 0]
            cy = self.start_pos[idx, 1]
            # 3 altitudes equally spaced across the vertical range, order randomized
            lo = 3.0
            hi = COURSE_H - 3.0
            spacing = (hi - lo) / 2.0  # gap between cylinders
            alts = np.array([
                lo,
                lo + spacing,
                lo + spacing * 2.0,
            ], dtype=np.float32)
            self.rng.shuffle(alts)
            for c in range(3):
                self.alt_cylinders[idx, c] = [cx, cy, alts[c]]

        # Yaw control state — start with random target heading
        self.yaw_hold_time[env_mask] = 0.0
        self.yaw_changes_done[env_mask] = 0
        self.target_yaw[env_mask] = self.rng.uniform(-np.pi, np.pi, n_reset).astype(np.float32)

        # Fly-to-point state — random target each episode for generalization
        self.fly_to_hold_time[env_mask] = 0.0
        for i in np.where(env_mask)[0]:
            self.fly_to_target[i] = np.array([
                self.rng.uniform(12.0, 28.0),
                self.rng.uniform(10.0, COURSE_D - 10.0),
                self.rng.uniform(3.0, 8.0),
            ], dtype=np.float32)

        # Ground thrash tracking
        self.ground_time[env_mask] = 0.0

        # Battery
        self.battery[env_mask] = 1.0

        # Wind — gated by curriculum level
        self.base_wind[env_mask] = 0.0
        self.turb_state[env_mask] = 0.0
        self.gust_remaining[env_mask] = 0.0

        wind_lo, wind_hi = self._wind_range
        wind_speed = self.rng.uniform(wind_lo, wind_hi, n_reset).astype(np.float32)
        wind_angle = self.rng.uniform(0.0, 2.0 * np.pi, n_reset).astype(np.float32)
        self.episode_wind[env_mask, 0] = wind_speed * np.cos(wind_angle)
        self.episode_wind[env_mask, 1] = wind_speed * np.sin(wind_angle)
        vert_scale = max(wind_hi * 0.15, 0.0)  # vertical wind proportional to horizontal
        self.episode_wind[env_mask, 2] = self.rng.uniform(
            -vert_scale, vert_scale, n_reset
        ).astype(np.float32)

        # Turbulence intensity — gated by curriculum level
        turb_lo, turb_hi = self._turb_range
        if turb_hi > 0:
            self.turb_intensity[env_mask] = self.rng.uniform(
                turb_lo, turb_hi, n_reset
            ).astype(np.float32)
        else:
            self.turb_intensity[env_mask] = 0.0

        # Projectiles
        self.proj_active[env_mask] = False
        self.proj_spawn_timer[env_mask] = PROJ_SPAWN_INTERVAL
        self.proj_source_turret[env_mask] = -1
        self._turret_flash.clear()

        # EW
        self.gps_denied[env_mask] = False
        self.jamming_intensity[env_mask] = 0.0

        # Domain randomization
        self._randomize_dynamics(env_mask, n_reset)

        # Compute initial waypoint distance
        if self.total_gates > 0:
            for i in np.where(env_mask)[0]:
                gi = min(self.current_gate[i], self.total_gates - 1)
                self.prev_waypoint_dist[i] = np.linalg.norm(
                    self.pos[i] - self.gates[gi]['center']
                )

    def _randomize_dynamics(self, env_mask, n_reset):
        """Domain randomization of physical parameters, scaled by curriculum level."""
        s = self._dr_scale  # 0.0 = nominal, 1.0 = full range
        if s < 0.01:
            # No randomization — use nominal values
            self.dr_mass[env_mask] = MASS
            self.dr_ixx[env_mask] = IXX
            self.dr_iyy[env_mask] = IYY
            self.dr_izz[env_mask] = IZZ
            self.dr_kf[env_mask] = KF
            self.dr_km[env_mask] = KM
            self.dr_motor_tau[env_mask] = MOTOR_TAU
            self.dr_drag_xy[env_mask] = DRAG_XY
            return
        self.dr_mass[env_mask] = self.rng.uniform(
            MASS - 0.1 * s, MASS + 0.15 * s, n_reset
        ).astype(np.float32)
        self.dr_ixx[env_mask] = self.rng.uniform(
            IXX * (1.0 - 0.2 * s), IXX * (1.0 + 0.2 * s), n_reset
        ).astype(np.float32)
        self.dr_iyy[env_mask] = self.dr_ixx[env_mask]
        self.dr_izz[env_mask] = self.rng.uniform(
            IZZ * (1.0 - 0.2 * s), IZZ * (1.0 + 0.2 * s), n_reset
        ).astype(np.float32)
        self.dr_kf[env_mask] = self.rng.uniform(
            KF * (1.0 - 0.2 * s), KF * (1.0 + 0.2 * s), n_reset
        ).astype(np.float32)
        self.dr_km[env_mask] = self.rng.uniform(
            KM * (1.0 - 0.2 * s), KM * (1.0 + 0.2 * s), n_reset
        ).astype(np.float32)
        self.dr_motor_tau[env_mask] = self.rng.uniform(
            max(0.01, MOTOR_TAU - 0.01 * s), MOTOR_TAU + 0.02 * s, n_reset
        ).astype(np.float32)
        self.dr_drag_xy[env_mask] = self.rng.uniform(
            max(0.005, DRAG_XY - 0.005 * s), DRAG_XY + 0.01 * s, n_reset
        ).astype(np.float32)

    # ══════════════════════════════════════════════════════════
    #  Physics Substep (Newton-Euler Rigid Body)
    # ══════════════════════════════════════════════════════════

    def _physics_substep(self):
        """Single physics substep at DT=0.005s. Vectorized across all envs."""
        N = self.n_envs
        mass = self.dr_mass                           # [N]
        kf = self.dr_kf                               # [N]
        km = self.dr_km                               # [N]
        motor_tau = self.dr_motor_tau                  # [N]
        ixx = self.dr_ixx                              # [N]
        iyy = self.dr_iyy
        izz = self.dr_izz

        # ── 1. Motor lag: first-order response ────────────────
        alpha = DT / motor_tau                         # [N]
        self.motor_omega += (self.motor_cmd - self.motor_omega) * alpha[:, np.newaxis]
        self.motor_omega = np.clip(self.motor_omega, 0.0, MAX_OMEGA)

        # ── 2. Thrust and torque from 4 motors ───────────────
        omega_sq = self.motor_omega ** 2               # [N, 4]
        thrusts = kf[:, np.newaxis] * omega_sq         # [N, 4] Newtons per motor

        # ── 2a. Ground effect (Cheeseman & Bennett 1955) ─────
        # IGE thrust boost: T_ige = T_oge / (1 - (R/(4*z))^2)
        # Significant below ~1 rotor diameter; fades above ~4R
        z = np.maximum(self.pos[:, 2], DRONE_RADIUS)   # height AGL
        ge_ratio = (PROP_RADIUS / (4.0 * z)) ** 2      # [N]
        ge_ratio = np.minimum(ge_ratio, 0.9)            # cap to avoid singularity
        ge_boost = 1.0 / (1.0 - ge_ratio)              # [N] multiplier ≥ 1.0
        thrusts *= ge_boost[:, np.newaxis]              # apply to each motor

        # Total thrust along body Z-axis
        total_thrust = thrusts.sum(axis=-1)            # [N]

        # X-config torques:
        #   Motor 0 (CW)  = front-right  (+x, +y)
        #   Motor 1 (CCW) = front-left   (-x, +y)
        #   Motor 2 (CW)  = back-left    (-x, -y)
        #   Motor 3 (CCW) = back-right   (+x, -y)
        tau_x = L_EFF * (thrusts[:, 1] + thrusts[:, 2] - thrusts[:, 0] - thrusts[:, 3])
        tau_y = L_EFF * (thrusts[:, 0] + thrusts[:, 1] - thrusts[:, 2] - thrusts[:, 3])
        tau_z = km[:, np.newaxis] * (
            omega_sq[:, 0:1] - omega_sq[:, 1:2] + omega_sq[:, 2:3] - omega_sq[:, 3:4]
        )
        tau_z = tau_z.squeeze(-1)                      # [N]

        # ── 3. Gyroscopic precession from spinning rotors ─────
        net_rotor_omega = (
            self.motor_omega[:, 0] - self.motor_omega[:, 1]
            + self.motor_omega[:, 2] - self.motor_omega[:, 3]
        )
        tau_x += J_MOTOR * self.ang_vel[:, 1] * net_rotor_omega
        tau_y -= J_MOTOR * self.ang_vel[:, 0] * net_rotor_omega

        # ── 4. Rotational drag ────────────────────────────────
        tau_x -= DRAG_ROT * self.ang_vel[:, 0]
        tau_y -= DRAG_ROT * self.ang_vel[:, 1]
        tau_z -= DRAG_ROT * self.ang_vel[:, 2]

        # ── 5. Euler's rotation equation: I*alpha = tau - omega x (I*omega) ──
        p = self.ang_vel[:, 0]
        q = self.ang_vel[:, 1]
        r = self.ang_vel[:, 2]
        alpha_x = (tau_x - (izz - iyy) * q * r) / ixx
        alpha_y = (tau_y - (ixx - izz) * p * r) / iyy
        alpha_z = (tau_z - (iyy - ixx) * p * q) / izz

        self.ang_vel[:, 0] += alpha_x * DT
        self.ang_vel[:, 1] += alpha_y * DT
        self.ang_vel[:, 2] += alpha_z * DT

        # Clamp angular velocity for numerical stability
        self.ang_vel = np.clip(self.ang_vel, -20.0, 20.0)

        # ── 6. Quaternion integration (Hamilton product) ──────
        qw = self.quat[:, 0]
        qx = self.quat[:, 1]
        qy = self.quat[:, 2]
        qz = self.quat[:, 3]
        wx = self.ang_vel[:, 0]
        wy = self.ang_vel[:, 1]
        wz = self.ang_vel[:, 2]
        hdt = 0.5 * DT
        self.quat[:, 0] += hdt * (-qx * wx - qy * wy - qz * wz)
        self.quat[:, 1] += hdt * (qw * wx + qy * wz - qz * wy)
        self.quat[:, 2] += hdt * (qw * wy - qx * wz + qz * wx)
        self.quat[:, 3] += hdt * (qw * wz + qx * wy - qy * wx)
        # Re-normalize to prevent quaternion drift
        qnorm = np.linalg.norm(self.quat, axis=-1, keepdims=True)
        self.quat /= (qnorm + 1e-10)

        # ── 7. Transform thrust to world frame ────────────────
        thrust_body = np.zeros((N, 3), dtype=np.float32)
        thrust_body[:, 2] = total_thrust
        thrust_world = self._quat_rotate(self.quat, thrust_body)

        # ── 8. Translational forces ───────────────────────────
        force = thrust_world.copy()
        force[:, 2] -= mass * GRAVITY                  # gravity

        # Aerodynamic drag (body-frame, then transform to world)
        vel_body = self._quat_rotate_inv(self.quat, self.vel)
        drag_body = np.zeros_like(vel_body)
        drag_xy = self.dr_drag_xy
        drag_body[:, 0] = -drag_xy * vel_body[:, 0]
        drag_body[:, 1] = -drag_xy * vel_body[:, 1]
        drag_body[:, 2] = -DRAG_Z * vel_body[:, 2]
        force += self._quat_rotate(self.quat, drag_body)

        # Wind disturbance (world frame, pre-computed)
        force += self.wind_force

        # ── 9. Ground normal force (applied before integration) ──
        # If the drone is on the ground and net force pushes it down,
        # the ground exerts a normal force to prevent penetration.
        on_ground = self.pos[:, 2] <= DRONE_RADIUS + 1e-4
        if np.any(on_ground):
            pushing_down = on_ground & (force[:, 2] < 0.0)
            force[pushing_down, 2] = 0.0  # ground cancels downward force

        # ── 10. Semi-implicit Euler integration ──────────────────
        accel = force / mass[:, np.newaxis]
        self.vel += accel * DT
        self.pos += self.vel * DT

        # ── 11. Ground collision (hard constraint) ───────────────
        # The ground is a rigid surface at z=0; the drone cannot penetrate.
        grounded = self.pos[:, 2] <= DRONE_RADIUS
        if np.any(grounded):
            # Track cumulative ground time (for thrash termination)
            self.ground_time[grounded] += DT
            # Reset ground time when airborne
            self.ground_time[~grounded] = 0.0
            # Clamp position to surface
            self.pos[grounded, 2] = DRONE_RADIUS
            # Kill all downward velocity (inelastic collision)
            self.vel[grounded, 2] = np.maximum(0.0, self.vel[grounded, 2])
            # Ground friction: strong damping on horizontal velocity
            self.vel[grounded, 0] *= 0.8
            self.vel[grounded, 1] *= 0.8
            # Damp angular velocity on ground contact (can't freely spin on floor)
            self.ang_vel[grounded] *= 0.5
        else:
            self.ground_time[:] = 0.0

        # ── 12. Store IMU readings ────────────────────────────
        # Real accelerometer measures: a_body + R^T * g (specific force)
        gravity_world = np.zeros((N, 3), dtype=np.float32)
        gravity_world[:, 2] = GRAVITY
        self.imu_accel = self._quat_rotate_inv(self.quat, accel + gravity_world)
        self.imu_gyro = self.ang_vel.copy()

        # Add sensor noise (domain randomization)
        imu_noise_std = 0.1
        self.imu_accel += self.rng.normal(0, imu_noise_std, (N, 3)).astype(np.float32)
        self.imu_gyro += self.rng.normal(0, imu_noise_std * 0.5, (N, 3)).astype(np.float32)

    # ══════════════════════════════════════════════════════════
    #  Wind Model (Dryden-based)
    # ══════════════════════════════════════════════════════════

    def _update_wind(self):
        """Update wind with Dryden-like turbulence + thermals + gusts."""
        N = self.n_envs

        # Base wind: mean-revert toward episode wind
        self.base_wind += (self.episode_wind - self.base_wind) * 0.01
        self.base_wind += self.rng.normal(0, 0.02, (N, 3)).astype(np.float32)

        # Dryden turbulence: first-order discrete filter
        # tau = L / V, where L = scale length (~10m horizontal, ~altitude vertical)
        V_ref = np.maximum(np.linalg.norm(self.vel, axis=-1, keepdims=True), 1.0)
        tau_uv = 10.0 / V_ref
        altitude = np.clip(self.pos[:, 2:3], 1.0, 50.0)
        tau_w = altitude / V_ref
        alpha_uv = np.exp(-RL_DT / tau_uv)
        alpha_w = np.exp(-RL_DT / tau_w)

        noise = self.rng.normal(0, 1, (N, 3)).astype(np.float32)
        sigma = self.turb_intensity[:, np.newaxis]
        self.turb_state[:, :2] = (
            alpha_uv * self.turb_state[:, :2]
            + np.sqrt(np.maximum(1.0 - alpha_uv ** 2, 0.0)) * sigma * noise[:, :2]
        )
        self.turb_state[:, 2:] = (
            alpha_w * self.turb_state[:, 2:]
            + np.sqrt(np.maximum(1.0 - alpha_w ** 2, 0.0)) * sigma * noise[:, 2:]
        )

        # Thermal updrafts
        updraft = np.zeros((N, 3), dtype=np.float32)
        for tz in self.thermal_zones:
            dx = self.pos[:, 0] - tz['x']
            dy = self.pos[:, 1] - tz['y']
            dist_sq = dx ** 2 + dy ** 2
            r2 = tz['radius'] ** 2
            strength = tz['strength'] * np.exp(-dist_sq / (2.0 * r2))
            updraft[:, 2] += strength
            edge = strength * 0.3
            updraft[:, 0] += self.rng.normal(0, 1, N).astype(np.float32) * edge
            updraft[:, 1] += self.rng.normal(0, 1, N).astype(np.float32) * edge

        # Wind gusts
        for i in range(N):
            if self.gust_remaining[i] > 0:
                updraft[i] += self.gust_dir[i] * self.gust_strength[i]
                self.gust_remaining[i] -= RL_DT
            elif self.rng.random() < 0.002:  # ~10% chance per second at 50Hz
                d = self.rng.normal(0, 1, 3).astype(np.float32)
                self.gust_dir[i] = d / (np.linalg.norm(d) + 1e-8)
                self.gust_strength[i] = self.rng.uniform(3.0, 8.0)
                self.gust_remaining[i] = self.rng.uniform(0.5, 2.0)

        # Obstacle wind interaction (shadow, venturi, wake turbulence)
        raw_wind = self.base_wind + self.turb_state + updraft
        raw_wind = self._apply_wind_obstacle_effects(raw_wind)

        self.wind_force = raw_wind * self.dr_mass[:, np.newaxis]

    def _apply_wind_obstacle_effects(self, wind):
        """Modify wind per-drone based on obstacle proximity and wind direction.

        Three effects:
        1. Wind shadow — obstacles upwind reduce wind (exponential recovery)
        2. Venturi — narrow gaps parallel to wind accelerate flow
        3. Wake turbulence — extra noise downwind of obstacles
        """
        if not self.obstacles:
            return wind

        N = self.n_envs
        wind_mag = np.linalg.norm(wind[:, :2], axis=-1, keepdims=True)  # [N, 1]
        # Skip if wind is negligible (avoids division by zero, saves cycles)
        weak = (wind_mag.squeeze() < 0.3)
        if weak.all():
            return wind

        # Wind direction (2D horizontal — vertical wind doesn't shadow)
        wind_dir = wind[:, :2] / np.maximum(wind_mag, 1e-6)  # [N, 2] unit vectors
        upwind_dir = -wind_dir  # direction toward wind source

        SHADOW_RANGE = 12.0   # max distance upwind to check for blockers
        SHADOW_DECAY = 0.15   # exponential recovery rate (1/meters)
        VENTURI_RANGE = 5.0   # lateral scan distance for gap detection
        VENTURI_BOOST = 1.4   # max wind multiplier in gaps
        WAKE_RANGE = 8.0      # downwind distance for wake turbulence
        WAKE_TURB = 0.6       # extra turbulence intensity in wake

        # Pre-compute per-drone: closest upwind obstacle distance & downwind distance
        shadow_factor = np.ones(N, dtype=np.float32)    # 1.0 = full wind, 0.3 = deep shadow
        venturi_factor = np.ones(N, dtype=np.float32)   # 1.0 = normal, up to VENTURI_BOOST
        wake_turb = np.zeros((N, 3), dtype=np.float32)  # additional turbulence

        drone_xy = self.pos[:, :2]  # [N, 2]
        drone_z = self.pos[:, 2]    # [N]

        for obs in self.obstacles:
            if obs['type'] == OBS_COLUMN:
                ox, oy, r = obs['x'], obs['y'], obs['radius']
                z_lo, z_hi = obs['z_min'], obs['z_max']
                # Vector from obstacle center to drone (2D)
                to_drone = drone_xy - np.array([ox, oy], dtype=np.float32)  # [N, 2]
                dist_xy = np.linalg.norm(to_drone, axis=-1)  # [N]
                in_height = (drone_z > z_lo - 1.0) & (drone_z < z_hi + 1.0)

                # How aligned is (obstacle→drone) with wind direction?
                # dot > 0 means drone is downwind of obstacle (wind shadow)
                to_drone_norm = to_drone / np.maximum(dist_xy[:, np.newaxis], 1e-6)
                alignment = np.sum(to_drone_norm * wind_dir, axis=-1)  # [N] -1..+1

                # --- Wind Shadow ---
                # Drone is downwind (alignment > 0.5) and within shadow cone
                shadow_mask = in_height & (alignment > 0.5) & (dist_xy < SHADOW_RANGE + r)
                if shadow_mask.any():
                    # Distance behind obstacle surface
                    d_behind = np.maximum(dist_xy - r, 0.1)
                    # Shadow strength: strongest right behind, exponential recovery
                    # Cross-section factor: narrower if drone is off-center
                    cross = np.clip((alignment - 0.5) * 2.0, 0.0, 1.0)  # 0..1
                    atten = 0.3 + 0.7 * (1.0 - np.exp(-SHADOW_DECAY * d_behind))
                    atten = np.where(shadow_mask, atten * cross + (1.0 - cross), 1.0)
                    shadow_factor = np.minimum(shadow_factor, atten)

                # --- Wake Turbulence ---
                wake_mask = in_height & (alignment > 0.3) & (dist_xy < WAKE_RANGE + r)
                if wake_mask.any():
                    d_behind_w = np.maximum(dist_xy - r, 0.1)
                    wake_str = WAKE_TURB * np.exp(-0.2 * d_behind_w) * np.clip(alignment, 0, 1)
                    wake_str = np.where(wake_mask, wake_str, 0.0)
                    wake_noise = self.rng.normal(0, 1, (N, 3)).astype(np.float32)
                    wake_turb += wake_noise * wake_str[:, np.newaxis]

                # --- Venturi (flanking check) ---
                # Drone is roughly beside the column (|alignment| < 0.4)
                # and close laterally — wind squeezes through gap
                flank_mask = in_height & (np.abs(alignment) < 0.4) & (dist_xy < r + VENTURI_RANGE)
                if flank_mask.any():
                    # Boost proportional to proximity
                    gap_factor = 1.0 + (VENTURI_BOOST - 1.0) * np.exp(-0.5 * (dist_xy - r))
                    gap_factor = np.where(flank_mask, np.clip(gap_factor, 1.0, VENTURI_BOOST), 1.0)
                    venturi_factor = np.maximum(venturi_factor, gap_factor)

            elif obs['type'] in (OBS_WALL, OBS_MOVING):
                mn = obs['min']
                mx = obs['max']
                center_xy = np.array([(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5], dtype=np.float32)
                half_ext = np.array([(mx[0] - mn[0]) * 0.5, (mx[1] - mn[1]) * 0.5], dtype=np.float32)
                z_lo, z_hi = mn[2], mx[2]

                to_drone = drone_xy - center_xy  # [N, 2]
                dist_xy = np.linalg.norm(to_drone, axis=-1)
                in_height = (drone_z > z_lo - 1.0) & (drone_z < z_hi + 1.0)

                to_drone_norm = to_drone / np.maximum(dist_xy[:, np.newaxis], 1e-6)
                alignment = np.sum(to_drone_norm * wind_dir, axis=-1)

                # Effective radius: project half-extents onto wind direction
                eff_r = np.abs(half_ext[0] * wind_dir[:, 0]) + np.abs(half_ext[1] * wind_dir[:, 1])

                # --- Wind Shadow ---
                shadow_mask = in_height & (alignment > 0.4) & (dist_xy < SHADOW_RANGE + eff_r)
                if shadow_mask.any():
                    d_behind = np.maximum(dist_xy - eff_r, 0.1)
                    cross = np.clip((alignment - 0.4) * 2.5, 0.0, 1.0)
                    atten = 0.3 + 0.7 * (1.0 - np.exp(-SHADOW_DECAY * d_behind))
                    atten = np.where(shadow_mask, atten * cross + (1.0 - cross), 1.0)
                    shadow_factor = np.minimum(shadow_factor, atten)

                # --- Wake Turbulence ---
                wake_mask = in_height & (alignment > 0.2) & (dist_xy < WAKE_RANGE + eff_r)
                if wake_mask.any():
                    d_behind_w = np.maximum(dist_xy - eff_r, 0.1)
                    wake_str = WAKE_TURB * np.exp(-0.2 * d_behind_w) * np.clip(alignment, 0, 1)
                    # Walls cast bigger wakes — scale by cross-section
                    wall_cross = np.minimum(eff_r, 3.0) / 1.5
                    wake_str = np.where(wake_mask, wake_str * wall_cross, 0.0)
                    wake_noise = self.rng.normal(0, 1, (N, 3)).astype(np.float32)
                    wake_turb += wake_noise * wake_str[:, np.newaxis]

                # --- Venturi ---
                flank_mask = in_height & (np.abs(alignment) < 0.3) & (dist_xy < eff_r + VENTURI_RANGE)
                if flank_mask.any():
                    gap_factor = 1.0 + (VENTURI_BOOST - 1.0) * np.exp(-0.5 * (dist_xy - eff_r))
                    gap_factor = np.where(flank_mask, np.clip(gap_factor, 1.0, VENTURI_BOOST), 1.0)
                    venturi_factor = np.maximum(venturi_factor, gap_factor)

        # Apply combined effects
        # Shadow reduces wind, venturi boosts it — shadow wins if both apply
        combined = shadow_factor * venturi_factor
        wind_modified = wind.copy()
        wind_modified[:, :2] *= combined[:, np.newaxis]
        wind_modified += wake_turb

        return wind_modified

    # ══════════════════════════════════════════════════════════
    #  LIDAR Raycasting
    # ══════════════════════════════════════════════════════════

    def _raycast_lidar(self):
        """Cast 12 rays from drone, return distances to nearest obstacle. [N, 12]"""
        N = self.n_envs
        distances = np.full((N, 12), MAX_LIDAR_RANGE, dtype=np.float32)

        # Transform ray directions from body frame to world frame
        for ri in range(12):
            ray_dir_body = LIDAR_DIRS[ri]  # [3]
            # Expand to [N, 3] and rotate
            ray_body_batch = np.tile(ray_dir_body, (N, 1))
            ray_world = self._quat_rotate(self.quat, ray_body_batch)  # [N, 3]

            for obs in self.obstacles:
                if obs['type'] in (OBS_WALL, OBS_MOVING):
                    t = self._ray_aabb(self.pos, ray_world, obs['min'], obs['max'])
                    distances[:, ri] = np.minimum(distances[:, ri], t)
                elif obs['type'] == OBS_COLUMN:
                    t = self._ray_cylinder(
                        self.pos, ray_world,
                        obs['x'], obs['y'], obs['radius'],
                        obs['z_min'], obs['z_max']
                    )
                    distances[:, ri] = np.minimum(distances[:, ri], t)

            # Floor and ceiling
            # Floor: plane z=0
            going_down = ray_world[:, 2] < -1e-6
            if going_down.any():
                t_f = np.full(N, MAX_LIDAR_RANGE, dtype=np.float32)
                t_f[going_down] = -self.pos[going_down, 2] / ray_world[going_down, 2]
                t_f = np.where(t_f > 0, t_f, MAX_LIDAR_RANGE)
                distances[:, ri] = np.minimum(distances[:, ri], t_f)

            # Ceiling: plane z=COURSE_H
            going_up = ray_world[:, 2] > 1e-6
            if going_up.any():
                t_c = np.full(N, MAX_LIDAR_RANGE, dtype=np.float32)
                t_c[going_up] = (COURSE_H - self.pos[going_up, 2]) / ray_world[going_up, 2]
                t_c = np.where(t_c > 0, t_c, MAX_LIDAR_RANGE)
                distances[:, ri] = np.minimum(distances[:, ri], t_c)

        # Add LIDAR noise
        lidar_noise = self.rng.normal(0, 0.05, (N, 12)).astype(np.float32)
        distances = np.maximum(0.0, distances + lidar_noise)

        self.lidar_distances = distances
        return distances

    @staticmethod
    def _ray_aabb(origins, dirs, box_min, box_max):
        """Vectorized ray-AABB intersection. Returns distances [N] (MAX_LIDAR_RANGE if no hit)."""
        N = origins.shape[0]
        inv_dir = 1.0 / (dirs + 1e-10)

        t1 = (box_min - origins) * inv_dir  # [N, 3]
        t2 = (box_max - origins) * inv_dir  # [N, 3]

        tmin = np.minimum(t1, t2)           # [N, 3]
        tmax = np.maximum(t1, t2)           # [N, 3]

        t_enter = tmin.max(axis=-1)         # [N]
        t_exit = tmax.min(axis=-1)          # [N]

        hit = (t_enter < t_exit) & (t_exit > 0)
        result = np.full(N, MAX_LIDAR_RANGE, dtype=np.float32)
        # Use t_enter if positive (ray starts outside), else t_exit (ray starts inside)
        t_hit = np.where(t_enter > 0, t_enter, t_exit)
        result[hit] = t_hit[hit]
        return result

    @staticmethod
    def _ray_cylinder(origins, dirs, cx, cy, radius, z_min, z_max):
        """Vectorized ray-cylinder intersection (infinite cylinder then clip Z)."""
        N = origins.shape[0]
        result = np.full(N, MAX_LIDAR_RANGE, dtype=np.float32)

        # Project to XY plane for circle intersection
        ox = origins[:, 0] - cx
        oy = origins[:, 1] - cy
        dx = dirs[:, 0]
        dy = dirs[:, 1]

        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - radius * radius

        discriminant = b * b - 4.0 * a * c
        has_hit = discriminant > 0

        if has_hit.any():
            sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
            t1 = (-b - sqrt_disc) / (2.0 * a + 1e-10)
            t2 = (-b + sqrt_disc) / (2.0 * a + 1e-10)

            # Check Z bounds for each t
            for t in [t1, t2]:
                z_at_t = origins[:, 2] + dirs[:, 2] * t
                valid = has_hit & (t > 0) & (z_at_t >= z_min) & (z_at_t <= z_max)
                result[valid] = np.minimum(result[valid], t[valid])

        return result

    # ══════════════════════════════════════════════════════════
    #  Projectile System
    # ══════════════════════════════════════════════════════════

    def _update_projectiles(self):
        """Spawn, advance, and detect projectile hits."""
        N = self.n_envs

        # Advance active projectiles
        for j in range(MAX_PROJECTILES):
            active = self.proj_active[:, j]
            if not active.any():
                continue
            self.proj_pos[active, j] += self.proj_vel[active, j] * RL_DT
            self.proj_vel[active, j, 2] -= GRAVITY * RL_DT  # gravity
            self.proj_life[active, j] -= RL_DT
            # Expire
            expired = active & (self.proj_life[:, j] <= 0)
            self.proj_active[expired, j] = False
            # Out of bounds
            oob = active & (
                (self.proj_pos[:, j, 2] < 0) |
                (np.abs(self.proj_pos[:, j, 0]) > COURSE_W + 10) |
                (np.abs(self.proj_pos[:, j, 1]) > COURSE_D + 10)
            )
            self.proj_active[oob, j] = False

        # Decay muzzle flashes
        for tid in list(self._turret_flash):
            self._turret_flash[tid] -= RL_DT
            if self._turret_flash[tid] <= 0:
                del self._turret_flash[tid]

        # Spawn timer — fire from turret installations
        self.proj_spawn_timer -= RL_DT
        spawn_mask = self.proj_spawn_timer <= 0
        if spawn_mask.any():
            has_turrets = len(self.turrets) > 0
            for i in np.where(spawn_mask)[0]:
                # Find inactive slot
                for j in range(MAX_PROJECTILES):
                    if not self.proj_active[i, j]:
                        if has_turrets:
                            # Round-robin turret selection
                            turret = self.turrets[self._turret_fire_idx % len(self.turrets)]
                            self._turret_fire_idx += 1
                            origin = turret['pos'].copy()
                            turret_id = turret['id']
                            # Muzzle flash for display env
                            if i == 0:
                                self._turret_flash[turret_id] = 0.15
                            self.proj_source_turret[i, j] = turret_id
                        else:
                            # Fallback: random spawn if no turrets (pre-13 levels)
                            angle = self.rng.uniform(0, 2.0 * np.pi)
                            spawn_dist = 20.0
                            origin = np.array([
                                self.pos[i, 0] + spawn_dist * np.cos(angle),
                                self.pos[i, 1] + spawn_dist * np.sin(angle),
                                self.pos[i, 2] + self.rng.uniform(-3.0, 3.0),
                            ], dtype=np.float32)
                            self.proj_source_turret[i, j] = -1
                        # Aim with lead targeting
                        to_target = self.pos[i] - origin
                        flight_time = np.linalg.norm(to_target) / PROJ_SPEED
                        lead_target = self.pos[i] + self.vel[i] * flight_time * self._turret_lead_factor
                        direction = lead_target - origin
                        direction /= (np.linalg.norm(direction) + 1e-8)
                        self.proj_pos[i, j] = origin
                        self.proj_vel[i, j] = direction * PROJ_SPEED
                        self.proj_active[i, j] = True
                        self.proj_life[i, j] = PROJ_LIFETIME
                        break
            interval = max(0.8, 2.5 - self._endless_difficulty * 0.35) if self._endless_mode else PROJ_SPAWN_INTERVAL
            self.proj_spawn_timer[spawn_mask] = interval + self.rng.uniform(
                -0.5, 0.5, spawn_mask.sum()
            ).astype(np.float32)

    def _check_projectile_hits(self):
        """Check if any projectile hit the drone. Returns [N] bool."""
        hit = np.zeros(self.n_envs, dtype=bool)
        for j in range(MAX_PROJECTILES):
            active = self.proj_active[:, j]
            if not active.any():
                continue
            dist = np.linalg.norm(self.pos - self.proj_pos[:, j], axis=-1)
            near_hit = active & (dist < DRONE_RADIUS + PROJ_RADIUS)
            hit |= near_hit
            self.proj_active[near_hit, j] = False
        return hit

    def _check_near_misses(self, threshold=1.0):
        """Count projectiles that passed within threshold but missed. Returns [N] float."""
        near_misses = np.zeros(self.n_envs, dtype=np.float32)
        for j in range(MAX_PROJECTILES):
            active = self.proj_active[:, j]
            if not active.any():
                continue
            dist = np.linalg.norm(self.pos - self.proj_pos[:, j], axis=-1)
            near = active & (dist < threshold) & (dist >= DRONE_RADIUS + PROJ_RADIUS)
            near_misses[near] += 1.0
        return near_misses

    # ══════════════════════════════════════════════════════════
    #  Electronic Warfare
    # ══════════════════════════════════════════════════════════

    def _update_ew(self):
        """Update electronic warfare state based on drone position."""
        self.gps_denied[:] = False
        for zone in self.gps_denial_zones:
            dist = np.linalg.norm(self.pos[:, :2] - zone['center'], axis=-1)
            self.gps_denied |= dist < zone['radius']

        self.jamming_intensity[:] = 0.0
        for zone in self.jamming_zones:
            dist = np.linalg.norm(self.pos[:, :2] - zone['center'], axis=-1)
            in_zone = dist < zone['radius']
            intensity = zone['intensity'] * (1.0 - dist / zone['radius'])
            self.jamming_intensity[in_zone] = np.maximum(
                self.jamming_intensity[in_zone], intensity[in_zone]
            )

    # ══════════════════════════════════════════════════════════
    #  Collision Detection
    # ══════════════════════════════════════════════════════════

    def _check_collisions(self):
        """Check drone-obstacle collisions. Returns [N] bool."""
        collided = np.zeros(self.n_envs, dtype=bool)

        for obs in self.obstacles:
            if obs['type'] in (OBS_WALL, OBS_MOVING):
                # Sphere vs AABB
                closest = np.clip(self.pos, obs['min'], obs['max'])
                dist = np.linalg.norm(self.pos - closest, axis=-1)
                collided |= dist < DRONE_RADIUS
            elif obs['type'] == OBS_COLUMN:
                dx = self.pos[:, 0] - obs['x']
                dy = self.pos[:, 1] - obs['y']
                dist_xy = np.sqrt(dx ** 2 + dy ** 2)
                in_height = (self.pos[:, 2] > obs['z_min']) & (self.pos[:, 2] < obs['z_max'])
                collided |= (dist_xy < obs['radius'] + DRONE_RADIUS) & in_height

        return collided

    def _check_gates(self):
        """Check if drone passed through its next gate. Returns [N] bool."""
        passed = np.zeros(self.n_envs, dtype=bool)

        for i in range(self.n_envs):
            gi = self.current_gate[i]
            if gi >= self.total_gates:
                continue
            gate = self.gates[gi]
            # Check if drone is near gate center and on the correct side
            rel = self.pos[i] - gate['center']
            # Distance along gate normal
            d_normal = np.dot(rel, gate['normal'])
            # Was the drone behind the gate last step? (we track via sign change)
            # Simplified: if within gate opening and close to gate plane
            d_right = np.dot(rel, gate['right'])
            d_up = np.dot(rel, gate['up'])
            in_opening = (
                abs(d_right) < gate['width'] / 2.0
                and abs(d_up) < gate['height'] / 2.0
            )
            close_to_plane = abs(d_normal) < 1.0  # within 1m of gate plane
            past_plane = d_normal > 0.0
            if in_opening and close_to_plane and past_plane:
                passed[i] = True

        return passed

    def _check_out_of_bounds(self):
        """Check if drone left course bounds. Returns [N] bool."""
        oob = (
            (self.pos[:, 0] < -2.0) | (self.pos[:, 0] > COURSE_W + 2.0) |
            (self.pos[:, 1] < -2.0) | (self.pos[:, 1] > COURSE_D + 2.0) |
            (self.pos[:, 2] > COURSE_H + 2.0)
        )
        return oob

    # ══════════════════════════════════════════════════════════
    #  Update Moving Obstacles
    # ══════════════════════════════════════════════════════════

    def _update_moving_obstacles(self):
        """Update positions of moving obstacles based on sim time."""
        t = self.sim_time[0]  # use env 0's time (they're all synced)
        for obs in self.obstacles:
            if obs['type'] == OBS_MOVING:
                offset = obs['amplitude'] * np.sin(2.0 * np.pi * t / obs['period'])
                ax = obs['axis']
                obs['min'] = obs['base_min'].copy()
                obs['max'] = obs['base_max'].copy()
                obs['min'][ax] += offset
                obs['max'][ax] += offset

    # ══════════════════════════════════════════════════════════
    #  Step
    # ══════════════════════════════════════════════════════════

    def step(self, actions: np.ndarray):
        """Run one RL step (4 physics substeps).

        Args:
            actions: [N, 1, ACTION_SIZE] — motor commands in [-1, 1]

        Returns:
            obs: [N, 1, OBS_SIZE]
            rewards: [N, 1]
            dones: [N] bool
            infos: dict
        """
        N = self.n_envs

        # Unpack actions: [N, 1, 4] → [N, 4]
        if actions.ndim == 3:
            act = actions[:, 0, :]
        else:
            act = actions

        # Store raw action for observation feedback (sim-to-real: motor lag awareness)
        self.prev_action = np.clip(act, -1.0, 1.0).astype(np.float32)

        # Map [-1, 1] → [0, MAX_OMEGA]
        self.prev_motor_cmd = self.motor_cmd.copy()
        self.motor_cmd = ((act + 1.0) / 2.0 * MAX_OMEGA).astype(np.float32)
        self.motor_cmd = np.clip(self.motor_cmd, 0.0, MAX_OMEGA)

        # Update wind (once per RL step)
        self._update_wind()

        # Update moving obstacles
        self._update_moving_obstacles()

        # Run physics substeps
        for _ in range(SUBSTEPS):
            self._physics_substep()

        # Update sim time
        self.sim_time += RL_DT
        self.step_count += 1

        # LIDAR
        self._raycast_lidar()

        # Projectiles (enabled by profile or endless mode D>=2)
        if self._endless_mode:
            profile = None
            if self._endless_difficulty >= 2.0:
                self._update_projectiles()
            if len(self.gps_denial_zones) > 0:
                self._update_ew()
        else:
            profile = CURRICULUM_PROFILES[self.curriculum_level]
            if profile.get('projectiles'):
                self._update_projectiles()
            if profile.get('ew'):
                self._update_ew()

        # Adversary drone (env 0 only, when enabled)
        if self._adversary_enabled:
            # Spawn after gate 2 is passed
            if not self.adv_active and self.current_gate[0] >= 2:
                self._spawn_adversary()
            self._update_adversary(RL_DT)
            if self._check_adversary_collision():
                if self._adversary_lethal:
                    # Apply projectile hit penalty and reset adversary
                    self.hits_taken[0] += 1
                self.adv_active = False  # despawn after hit

        # ── Rewards ───────────────────────────────────────────
        rewards = np.zeros(N, dtype=np.float32)

        # Survival
        rewards += self.survival_reward

        # Waypoint progress (potential-based)
        for i in range(N):
            gi = self.current_gate[i]
            if gi < self.total_gates:
                curr_dist = np.linalg.norm(self.pos[i] - self.gates[gi]['center'])
                progress = self.prev_waypoint_dist[i] - curr_dist
                rewards[i] += progress * self.progress_reward
                self.prev_waypoint_dist[i] = curr_dist

        # Gate passage + alignment bonus
        passed = self._check_gates()
        rewards[passed] += self.waypoint_reward
        # Gate alignment bonus: reward perpendicular approach
        for i in np.where(passed)[0]:
            gi = self.current_gate[i]  # still pointing at the gate we just passed
            if gi < self.total_gates:
                gate = self.gates[gi]
                # Alignment = |dot(velocity_dir, gate_normal)|, 1.0 = perfect perpendicular
                vel_mag = np.linalg.norm(self.vel[i])
                if vel_mag > 0.5:
                    vel_dir = self.vel[i] / vel_mag
                    alignment = abs(np.dot(vel_dir, gate['normal']))
                    rewards[i] += alignment * self.gate_align_bonus
        self.current_gate[passed] += 1
        # Update waypoint distance for newly targeted gate
        for i in np.where(passed)[0]:
            gi = self.current_gate[i]
            if gi < self.total_gates:
                self.prev_waypoint_dist[i] = np.linalg.norm(
                    self.pos[i] - self.gates[gi]['center']
                )

        # ── Completion (mode-dependent) ─────────────────────────
        tilt = self._compute_tilt_angle()  # [N] radians — needed for hover check too
        alt = self.pos[:, 2]

        if self._completion_mode == 'gates':
            # Standard: all gates passed
            completed = self.current_gate >= self.total_gates
            rewards[completed] += self.course_complete_reward

        elif self._completion_mode == 'hover':
            # Hover: stay near start pos + in altitude band + level
            # Box shrinks each time the drone holds for HOVER_TARGET_DURATION
            h_drift = np.linalg.norm(
                self.pos[:, :2] - self.start_pos[:, :2], axis=-1
            )
            # Current zone radius shrinks by HOVER_SHRINK_FACTOR per stage
            shrink = HOVER_SHRINK_FACTOR ** self.hover_shrink_stage
            cur_radius = HOVER_POS_RADIUS * shrink
            alt_mid = (ALTITUDE_TARGET_LO + ALTITUDE_TARGET_HI) / 2.0
            alt_half = (ALTITUDE_TARGET_HI - ALTITUDE_TARGET_LO) / 2.0 * shrink
            cur_alt_lo = alt_mid - alt_half
            cur_alt_hi = alt_mid + alt_half

            in_hold = (
                (alt >= cur_alt_lo) &
                (alt <= cur_alt_hi) &
                (tilt < HOVER_MAX_TILT) &
                (h_drift < cur_radius)
            )
            self.hover_duration[in_hold] += RL_DT
            self.hover_duration[~in_hold] = np.maximum(
                0.0, self.hover_duration[~in_hold] - RL_DT * 0.5
            )

            # ── Per-step hover zone bonus: "being inside box = good" ──
            rewards[in_hold] += 0.05

            # ── Per-stage hold duration: last stage requires 10s ──
            is_final_stage = (self.hover_shrink_stage == self._hover_stages - 1)
            hold_required = np.where(is_final_stage, 10.0, HOVER_TARGET_DURATION)

            # ── Hover progress shaping: proportional to hold progress ──
            progress = self.hover_duration / hold_required
            rewards += np.clip(progress, 0.0, 1.0) * 0.03

            # ── Velocity damping: encourage stillness ──
            # Unlike speed_coeff (only > 10 m/s), this penalizes ALL velocity.
            # Keep coefficient gentle to avoid teaching "don't move at all".
            vel_mag = np.linalg.norm(self.vel, axis=-1)
            rewards -= vel_mag * 0.008

            # ── Target altitude: Gaussian peak at zone midpoint ──
            # The band-based altitude reward is flat within 2-8m so the drone
            # doesn't know WHERE to hover. This adds a gradient toward center.
            alt_err = np.abs(alt - alt_mid)
            rewards += np.exp(-alt_err ** 2 / 4.0) * 0.04

            # Check which envs passed the current stage
            stage_passed = self.hover_duration >= hold_required
            if stage_passed.any():
                rewards[stage_passed] += LANDING_REWARD
                # Advance to next shrink stage, reset hold timer
                not_final = stage_passed & (self.hover_shrink_stage < self._hover_stages)
                self.hover_shrink_stage[not_final] += 1
                self.hover_duration[stage_passed] = 0.0

            completed = self.hover_shrink_stage >= self._hover_stages

        elif self._completion_mode == 'takeoff':
            # Takeoff: reach altitude band from ground, hold for N seconds
            h_drift = np.linalg.norm(
                self.pos[:, :2] - self.start_pos[:, :2], axis=-1
            )
            in_band = (
                (alt >= TAKEOFF_TARGET_ALT) &
                (tilt < HOVER_MAX_TILT) &
                (h_drift < HOVER_POS_RADIUS * 2.0)  # more lenient drift
            )
            self.takeoff_hold_time[in_band] += RL_DT
            self.takeoff_hold_time[~in_band] = np.maximum(
                0.0, self.takeoff_hold_time[~in_band] - RL_DT
            )
            completed = self.takeoff_hold_time >= TAKEOFF_HOLD_DURATION
            rewards[completed] += LANDING_REWARD

        elif self._completion_mode == 'land':
            # Landing: gentle touchdown (low vel, near ground)
            on_ground = alt <= DRONE_RADIUS + 0.05
            low_vel = np.linalg.norm(self.vel, axis=-1) < LANDING_MAX_VEL
            gentle_land = on_ground & low_vel
            completed = gentle_land
            rewards[gentle_land] += LANDING_REWARD

        elif self._completion_mode == 'altitude_change':
            # Altitude change: fly to 3 cylinders in sequence, hold inside each
            # Each cylinder has a 3D center; drone must be within radius + height
            cyl_idx = self.alt_current_cyl  # [N] int
            # Gather current cylinder center for each env
            cyl_center = np.array([self.alt_cylinders[i, cyl_idx[i]] for i in range(N)])
            to_cyl = cyl_center - self.pos  # [N, 3]
            horiz_dist = np.linalg.norm(to_cyl[:, :2], axis=-1)  # XY distance
            vert_err = np.abs(to_cyl[:, 2])  # Z error
            inside_cyl = (horiz_dist < ALT_CYL_RADIUS) & (vert_err < ALT_CYL_HEIGHT) & (tilt < HOVER_MAX_TILT)
            self.alt_hold_time[inside_cyl] += RL_DT
            self.alt_hold_time[~inside_cyl] = np.maximum(
                0.0, self.alt_hold_time[~inside_cyl] - RL_DT
            )
            # Reward: progress toward current cylinder + proximity bonus inside
            total_dist = np.sqrt(horiz_dist ** 2 + to_cyl[:, 2] ** 2)
            rewards += np.maximum(0.0, 1.0 - total_dist / 10.0) * self.altitude_coeff
            # Check if hold duration reached — advance to next cylinder
            cyl_done = self.alt_hold_time >= ALT_CHANGE_HOLD
            if np.any(cyl_done):
                self.alt_changes_done[cyl_done] += 1
                self.alt_current_cyl[cyl_done] = np.minimum(
                    self.alt_current_cyl[cyl_done] + 1, 2
                )
                self.alt_hold_time[cyl_done] = 0.0
                rewards[cyl_done] += LANDING_REWARD * 0.5  # partial completion bonus
            completed = self.alt_changes_done >= 3

        elif self._completion_mode == 'yaw':
            # Yaw control: rotate to target heading, hold, then get new target
            # Complete after 3 successful yaw changes
            euler = self._quat_to_euler(self.quat)
            yaw = euler[:, 2]  # current yaw angle
            yaw_err = np.abs(np.arctan2(np.sin(yaw - self.target_yaw),
                                        np.cos(yaw - self.target_yaw)))
            at_target = (yaw_err < YAW_TOLERANCE) & (tilt < HOVER_MAX_TILT)
            self.yaw_hold_time[at_target] += RL_DT
            self.yaw_hold_time[~at_target] = np.maximum(
                0.0, self.yaw_hold_time[~at_target] - RL_DT
            )
            # Reward yaw proximity — dominant signal guiding rotation
            rewards += np.maximum(0.0, 1.0 - yaw_err / np.pi) * 0.04
            # Extra bonus when within tolerance (encourage holding)
            rewards[at_target] += 0.02
            # Containment zone — penalize drifting outside the pirouette zone
            h_drift_yaw = np.linalg.norm(
                self.pos[:, :2] - self.start_pos[:, :2], axis=-1
            )
            outside = h_drift_yaw > YAW_ZONE_RADIUS
            # Penalty proportional to distance outside the zone boundary
            rewards[outside] -= (h_drift_yaw[outside] - YAW_ZONE_RADIUS) * 0.1
            # Check hold
            yaw_done = self.yaw_hold_time >= YAW_TARGET_HOLD
            if np.any(yaw_done):
                self.yaw_changes_done[yaw_done] += 1
                self.yaw_hold_time[yaw_done] = 0.0
                for i in np.where(yaw_done)[0]:
                    self.target_yaw[i] = self.rng.uniform(-np.pi, np.pi)
                rewards[yaw_done] += LANDING_REWARD * 0.5
            completed = self.yaw_changes_done >= 3

        elif self._completion_mode == 'fly_to_point':
            # Fly to point: navigate to target position, hover there briefly
            dist_to_target = np.linalg.norm(self.pos - self.fly_to_target, axis=-1)
            near_target = (dist_to_target < FLY_TO_RADIUS) & (tilt < HOVER_MAX_TILT)
            self.fly_to_hold_time[near_target] += RL_DT
            self.fly_to_hold_time[~near_target] = np.maximum(
                0.0, self.fly_to_hold_time[~near_target] - RL_DT
            )
            # Progress reward toward target
            rewards += np.maximum(0.0, 1.0 - dist_to_target / 30.0) * self.progress_reward
            completed = self.fly_to_hold_time >= FLY_TO_HOLD
            rewards[completed] += LANDING_REWARD

        else:
            completed = np.zeros(N, dtype=bool)

        # ── Per-step rewards ────────────────────────────────────

        # Drift penalty (horizontal distance from start position)
        if self.drift_coeff > 0:
            h_drift = np.linalg.norm(
                self.pos[:, :2] - self.start_pos[:, :2], axis=-1
            )
            rewards -= h_drift * self.drift_coeff

        # Stability penalty (excessive tilt + angular velocity)
        ang_vel_mag = np.linalg.norm(self.ang_vel, axis=-1)
        rewards -= (tilt * 0.5 + ang_vel_mag * 0.5) * self.stability_coeff

        # Orientation penalty (inverted = qw near 0)
        # qw=1 is upright, qw=0 is 180-deg tilt. Penalize 1-qw^2.
        qw_sq = self.quat[:, 0] ** 2
        rewards -= (1.0 - qw_sq) * self.orientation_coeff

        # Energy penalty (motor usage)
        motor_usage = np.mean(self.motor_omega / MAX_OMEGA, axis=-1)
        rewards -= motor_usage * self.energy_coeff

        # Smoothness penalty (motor command jerk)
        motor_delta = np.abs(self.motor_cmd - self.prev_motor_cmd) / MAX_OMEGA
        rewards -= np.mean(motor_delta, axis=-1) * self.smoothness_coeff

        # Altitude hold reward (prefer flying in target band)
        # Skip for altitude_change — cylinder proximity reward handles altitude there,
        # and the 2-8m band would penalize flying to cylinders above 8m
        if self._completion_mode != 'altitude_change':
            in_band = (alt >= ALTITUDE_TARGET_LO) & (alt <= ALTITUDE_TARGET_HI)
            below = alt < ALTITUDE_TARGET_LO
            above = alt > ALTITUDE_TARGET_HI
            alt_reward = np.zeros(N, dtype=np.float32)
            alt_reward[in_band] = 1.0
            alt_reward[below] = np.clip(alt[below] / ALTITUDE_TARGET_LO, 0.0, 1.0)
            alt_reward[above] = np.clip(1.0 - (alt[above] - ALTITUDE_TARGET_HI) / ALTITUDE_TARGET_HI, 0.0, 1.0)
            rewards += alt_reward * self.altitude_coeff

        # Speed penalty (excessive velocity)
        speed = np.linalg.norm(self.vel, axis=-1)
        excess_speed = np.maximum(0.0, speed - SPEED_LIMIT)
        rewards -= excess_speed * self.speed_coeff

        # Collision
        collided = self._check_collisions()
        rewards[collided] += self.collision_penalty

        # Projectile hits
        hit = self._check_projectile_hits()
        rewards[hit] += self.projectile_hit_penalty
        self.hits_taken[hit] += 1

        # Near-miss dodge bonus
        near_misses = self._check_near_misses(threshold=1.0)
        rewards += near_misses * self.dodge_bonus

        # Projectile avoidance shaping — reward for keeping distance from active projectiles
        if self.proj_avoid_bonus > 0:
            for j in range(MAX_PROJECTILES):
                active = self.proj_active[:, j]
                if not active.any():
                    continue
                dist = np.linalg.norm(self.pos - self.proj_pos[:, j], axis=-1)
                # Reward for being >3m from projectile, scaled by proximity
                avoid_r = np.where(active, np.clip(dist / 5.0, 0.0, 1.0), 0.0)
                rewards += avoid_r * self.proj_avoid_bonus

        # Out of bounds
        oob = self._check_out_of_bounds()
        rewards[oob] += self.oob_penalty

        # Hard crash (ground impact with high vertical speed)
        # In landing mode, ground contact is the goal — only penalize fast impacts
        near_ground = self.pos[:, 2] <= DRONE_RADIUS + 0.01
        if self._completion_mode == 'land':
            # Landing: only crash if descent is too fast (> LANDING_MAX_VEL)
            hard_crash = near_ground & (self.vel[:, 2] < -LANDING_MAX_VEL * 2.0)
        else:
            hard_crash = near_ground & (self.vel[:, 2] < -5.0)
        rewards[hard_crash] += self.crash_penalty

        # Battery drain
        power = np.sum(self.dr_kf[:, np.newaxis] * self.motor_omega ** 3, axis=-1)
        hover_power = np.sum(self.dr_kf * HOVER_OMEGA ** 3) * np.ones(N)
        drain_rate = power / (hover_power + 1e-8) / BATTERY_CAPACITY
        self.battery -= drain_rate * RL_DT
        self.battery = np.clip(self.battery, 0.0, 1.0)
        battery_dead = self.battery <= 0.0

        # Ground thrash termination — end episode if stuck on ground for 2s
        # (Does not apply to ground-start modes before liftoff)
        ground_thrash = self.ground_time >= 2.0
        if self._start_mode == 'ground':
            # Only penalize if the drone has previously been airborne (step > 50)
            ground_thrash &= (self.step_count >= 50)
        rewards[ground_thrash] += self.crash_penalty

        # Timeout penalty — punish running out the clock on gate courses
        timed_out = (self.step_count >= self.max_steps) & ~completed
        if self._completion_mode == 'gates' and self.timeout_penalty < 0:
            rewards[timed_out] += self.timeout_penalty

        # ── Done conditions ───────────────────────────────────
        # Any collision or hit is fatal — episode ends immediately
        dones = collided | hit | oob | hard_crash | completed | battery_dead | ground_thrash
        dones |= (self.step_count >= self.max_steps)

        # ── Track curriculum ──────────────────────────────────
        endless_advanced = False
        for i in np.where(dones)[0]:
            self.recent_completions.append(int(completed[i]))
            if len(self.recent_completions) > self.curriculum_window:
                self.recent_completions = self.recent_completions[-self.curriculum_window:]

        # Endless mode: completion of any env triggers next scenario for ALL envs
        if self._endless_mode and completed.any():
            self._advance_endless_scenario()
            endless_advanced = True

        # Endless mode: failure resets streak
        if self._endless_mode and dones.any() and not completed.any():
            self._endless_current_streak = 0

        # Auto-reset done environments (skip if endless advanced — already reset all)
        if not endless_advanced:
            done_mask = dones.copy()
            if done_mask.any():
                self.reset(done_mask)
        else:
            done_mask = dones.copy()

        # ── Build observations ────────────────────────────────
        obs = self.get_obs()

        # ── Infos ─────────────────────────────────────────────
        infos = {
            'episode_done': done_mask,
            'collided': collided,
            'completed': completed & done_mask,
            'oob': oob,
            'gates_passed': self.current_gate.copy(),
            'battery': self.battery.copy(),
        }

        # Reshape rewards for PPO: [N] → [N, 1]
        rewards_out = rewards[:, np.newaxis]

        return obs, rewards_out, dones, infos

    def _compute_tilt_angle(self):
        """Compute tilt angle from vertical (radians). [N]"""
        # Body Z-axis in world frame
        body_up = np.zeros((self.n_envs, 3), dtype=np.float32)
        body_up[:, 2] = 1.0
        world_up = self._quat_rotate(self.quat, body_up)
        # Angle between world_up and [0,0,1]
        cos_angle = np.clip(world_up[:, 2], -1.0, 1.0)
        return np.arccos(cos_angle)

    # ══════════════════════════════════════════════════════════
    #  Observations
    # ══════════════════════════════════════════════════════════

    def get_obs(self):
        """Build observation tensor. Returns [N, 1, OBS_SIZE]."""
        N = self.n_envs
        if self._endless_mode:
            profile = {'ew': len(self.gps_denial_zones) > 0}
        else:
            profile = CURRICULUM_PROFILES[self.curriculum_level]
        obs = np.zeros((N, OBS_SIZE), dtype=np.float32)

        # 0-2: IMU accelerometer (body frame, normalized)
        obs[:, 0:3] = self.imu_accel / 20.0

        # 3-5: IMU gyroscope (body frame, normalized)
        obs[:, 3:6] = self.imu_gyro / 10.0

        # 6-9: Orientation quaternion
        obs[:, 6:10] = self.quat

        # 10-12: Velocity in body frame (normalized)
        vel_body = self._quat_rotate_inv(self.quat, self.vel)
        obs[:, 10:13] = vel_body / 15.0

        # 13: Altitude (normalized)
        obs[:, 13] = self.pos[:, 2] / COURSE_H

        # 14: Vertical velocity (normalized)
        obs[:, 14] = self.vel[:, 2] / 10.0

        # 15-18: Motor RPMs (normalized)
        obs[:, 15:19] = self.motor_omega / MAX_OMEGA

        # 19-30: LIDAR distances (normalized)
        obs[:, 19:31] = self.lidar_distances / MAX_LIDAR_RANGE

        # 31-33: Target bearing (body frame unit vector)
        # 34: Target distance (normalized)
        # For gates mode: bearing to next gate. For fly_to_point: bearing to target.
        # For altitude_change: vertical error direction. For yaw: yaw error.
        if self._completion_mode == 'gates':
            for i in range(N):
                gi = self.current_gate[i]
                if gi < self.total_gates:
                    to_gate = self.gates[gi]['center'] - self.pos[i]
                    dist = np.linalg.norm(to_gate)
                    if dist > 0.01:
                        bearing_world = to_gate / dist
                        bearing_body = self._quat_rotate_inv(
                            self.quat[i:i+1], bearing_world[np.newaxis]
                        )[0]
                        obs[i, 31:34] = bearing_body
                    obs[i, 34] = dist / max(COURSE_W, COURSE_D)
        elif self._completion_mode == 'fly_to_point':
            to_target = self.fly_to_target - self.pos
            dist = np.linalg.norm(to_target, axis=-1, keepdims=True)
            safe_dist = np.maximum(dist, 0.01)
            bearing_world = to_target / safe_dist
            bearing_body = self._quat_rotate_inv(self.quat, bearing_world)
            obs[:, 31:34] = bearing_body
            obs[:, 34] = dist.squeeze(-1) / max(COURSE_W, COURSE_D)
        elif self._completion_mode == 'altitude_change':
            # Encode bearing to current cylinder center (3D)
            cyl_idx = self.alt_current_cyl  # [N]
            cyl_center = np.array([self.alt_cylinders[i, cyl_idx[i]] for i in range(N)])
            to_cyl = cyl_center - self.pos  # [N, 3]
            dist = np.linalg.norm(to_cyl, axis=-1, keepdims=True)
            safe_dist = np.maximum(dist, 0.01)
            bearing_world = to_cyl / safe_dist
            bearing_body = self._quat_rotate_inv(self.quat, bearing_world)
            obs[:, 31:34] = bearing_body
            obs[:, 34] = dist.squeeze(-1) / COURSE_H
        elif self._completion_mode == 'yaw':
            # Encode target yaw as bearing direction
            euler = self._quat_to_euler(self.quat)
            yaw_err = np.arctan2(np.sin(self.target_yaw - euler[:, 2]),
                                 np.cos(self.target_yaw - euler[:, 2]))
            obs[:, 31] = np.sin(yaw_err)  # body-frame heading error X
            obs[:, 32] = np.cos(yaw_err)  # body-frame heading error Y
            obs[:, 33] = 0.0
            obs[:, 34] = np.abs(yaw_err) / np.pi
        elif self._completion_mode in ('hover', 'takeoff'):
            # Encode drift from start position so the drone can correct
            # Without this, the drone has no way to know it has drifted
            to_start = self.start_pos - self.pos  # world-frame vector back to start
            dist = np.linalg.norm(to_start[:, :2], axis=-1, keepdims=True)  # horizontal only
            safe_dist = np.maximum(dist, 0.01)
            # Normalize horizontal component to unit vector
            bearing_world = np.zeros((N, 3), dtype=np.float32)
            bearing_world[:, :2] = to_start[:, :2] / safe_dist
            # Also encode altitude error toward target midpoint
            alt_mid = (ALTITUDE_TARGET_LO + ALTITUDE_TARGET_HI) / 2.0
            bearing_world[:, 2] = np.clip((alt_mid - self.pos[:, 2]) / 5.0, -1.0, 1.0)
            bearing_body = self._quat_rotate_inv(self.quat, bearing_world)
            obs[:, 31:34] = bearing_body
            obs[:, 34] = dist.squeeze(-1) / max(COURSE_W, COURSE_D)

        # 35: Waypoint/task progress
        if self._completion_mode == 'gates' and self.total_gates > 0:
            obs[:, 35] = self.current_gate.astype(np.float32) / self.total_gates
        elif self._completion_mode == 'altitude_change':
            obs[:, 35] = self.alt_changes_done.astype(np.float32) / 3.0
        elif self._completion_mode == 'yaw':
            obs[:, 35] = self.yaw_changes_done.astype(np.float32) / 3.0
        elif self._completion_mode == 'fly_to_point':
            obs[:, 35] = np.clip(self.fly_to_hold_time / FLY_TO_HOLD, 0.0, 1.0)
        elif self._completion_mode in ('hover', 'takeoff'):
            obs[:, 35] = np.clip(self.hover_duration / 2.0, 0.0, 1.0)
        else:
            obs[:, 35] = 0.0

        # 36: Step fraction
        obs[:, 36] = self.step_count.astype(np.float32) / self.max_steps

        # 37-39: Wind estimate (body frame, from accel residuals)
        # In reality, estimated by comparing expected vs actual acceleration
        # We give a noisy version of the actual wind
        wind_body = self._quat_rotate_inv(self.quat, self.base_wind + self.turb_state)
        wind_noise = self.rng.normal(0, 0.5, (N, 3)).astype(np.float32)
        obs[:, 37:40] = (wind_body + wind_noise) / 10.0

        # 40: GPS available flag
        obs[:, 40] = (~self.gps_denied).astype(np.float32)

        # 41: Sensor health
        # Under jamming, some sensors degrade
        obs[:, 41] = 1.0 - self.jamming_intensity * 0.5

        # 42: Battery fraction
        obs[:, 42] = self.battery

        # 43-45: Nearest threat bearing (body frame)
        for i in range(N):
            min_dist = float('inf')
            nearest_dir = np.zeros(3, dtype=np.float32)
            for j in range(MAX_PROJECTILES):
                if self.proj_active[i, j]:
                    d = np.linalg.norm(self.proj_pos[i, j] - self.pos[i])
                    if d < min_dist:
                        min_dist = d
                        direction = self.proj_pos[i, j] - self.pos[i]
                        nearest_dir = direction / (d + 1e-8)
            if min_dist < float('inf'):
                bearing_body = self._quat_rotate_inv(
                    self.quat[i:i+1], nearest_dir[np.newaxis]
                )[0]
                obs[i, 43:46] = bearing_body

        # 46-49: Previous action (raw [-1,1] motor commands from last step)
        # Critical for sim-to-real: real motors have lag (MOTOR_TAU), so the
        # policy needs to know what it already commanded to anticipate response.
        # Every real flight controller tracks its own output history.
        obs[:, 46:50] = self.prev_action

        # Apply EW effects to observations
        if profile.get('ew'):
            # GPS denial: zero out waypoint bearing/distance
            denied = self.gps_denied
            obs[denied, 31:35] = 0.0
            obs[denied, 40] = 0.0

            # Sensor jamming: add noise
            jamming = self.jamming_intensity
            noise = self.rng.normal(0, 1, (N, OBS_SIZE)).astype(np.float32)
            obs += noise * jamming[:, np.newaxis] * 0.3

            # Sensor dropout (random channels zeroed)
            dropout_prob = 0.01 + jamming * 0.15
            dropout_mask = (self.rng.random((N, OBS_SIZE)) > dropout_prob[:, np.newaxis]).astype(np.float32)
            obs *= dropout_mask

        # Clip to reasonable range
        obs = np.clip(obs, -5.0, 5.0)

        # Return as [N, 1, OBS_SIZE] for PPO compatibility (1 agent per env)
        return obs[:, np.newaxis, :]

    # ══════════════════════════════════════════════════════════
    #  Course Layout (sent once to new clients)
    # ══════════════════════════════════════════════════════════

    def get_course_layout(self) -> dict:
        """Serialize the full course geometry for Three.js rendering.

        Sent once on WS connect and whenever curriculum level changes.
        """
        obs_out = []
        for obs in self.obstacles:
            if obs['type'] in (OBS_WALL, OBS_MOVING):
                entry = {
                    'type': 'wall' if obs['type'] == OBS_WALL else 'moving',
                    'min': [float(obs['min'][k]) for k in range(3)],
                    'max': [float(obs['max'][k]) for k in range(3)],
                    'sector': obs.get('sector', 1),
                }
                if obs['type'] == OBS_MOVING:
                    entry['base_min'] = [float(obs['base_min'][k]) for k in range(3)]
                    entry['base_max'] = [float(obs['base_max'][k]) for k in range(3)]
                    entry['axis'] = obs['axis']
                    entry['amplitude'] = float(obs['amplitude'])
                    entry['period'] = float(obs['period'])
                obs_out.append(entry)
            elif obs['type'] == OBS_COLUMN:
                obs_out.append({
                    'type': 'column',
                    'x': float(obs['x']),
                    'y': float(obs['y']),
                    'radius': float(obs['radius']),
                    'z_min': float(obs['z_min']),
                    'z_max': float(obs['z_max']),
                    'sector': obs.get('sector', 1),
                })

        gates_out = []
        for g in self.gates:
            gates_out.append({
                'x': float(g['center'][0]),
                'y': float(g['center'][1]),
                'z': float(g['center'][2]),
                'width': float(g['width']),
                'height': float(g['height']),
                'nx': float(g['normal'][0]),
                'ny': float(g['normal'][1]),
                'nz': float(g['normal'][2]),
                'sector': g.get('sector', 1),
            })

        thermals_out = []
        for tz in self.thermal_zones:
            thermals_out.append({
                'x': float(tz['x']),
                'y': float(tz['y']),
                'radius': float(tz['radius']),
                'strength': float(tz['strength']),
            })

        ew_out = {
            'gps_denial': [{'x': float(z['center'][0]), 'y': float(z['center'][1]),
                           'radius': float(z['radius'])} for z in self.gps_denial_zones],
            'jamming': [{'x': float(z['center'][0]), 'y': float(z['center'][1]),
                        'radius': float(z['radius']), 'intensity': float(z['intensity'])}
                       for z in self.jamming_zones],
        }

        if self._endless_mode:
            level_name = f'Endless #{self._endless_scenario}'
            completion_mode = 'gates'
        else:
            profile = CURRICULUM_PROFILES[self.curriculum_level]
            level_name = profile.get('name', '')
            completion_mode = profile.get('completion_mode', 'gates')

        turrets_out = []
        for t in self.turrets:
            turrets_out.append({
                'id': t['id'],
                'x': float(t['pos'][0]),
                'y': float(t['pos'][1]),
                'z': float(t['pos'][2]),
            })

        layout = {
            'type': 'course_layout',
            'curriculum_level': self.curriculum_level,
            'level_name': level_name,
            'completion_mode': completion_mode,
            'total_gates': self.total_gates,
            'obstacles': obs_out,
            'gates': gates_out,
            'thermals': thermals_out,
            'ew_zones': ew_out,
            'turrets': turrets_out,
            'bounds': {
                'w': COURSE_W, 'd': COURSE_D, 'h': COURSE_H,
            },
        }
        if self._endless_mode:
            layout['endless'] = {
                'scenario': self._endless_scenario,
                'difficulty': round(self._endless_difficulty, 2),
            }
        return layout

    # ══════════════════════════════════════════════════════════
    #  Display State (for Three.js visualization)
    # ══════════════════════════════════════════════════════════

    def get_display_state(self, env_idx: int = 0) -> dict:
        """Serialize state of one environment for WebSocket broadcast."""
        i = env_idx
        euler = self._quat_to_euler(self.quat[i:i+1])[0]

        # Drone state
        drone = {
            'x': float(self.pos[i, 0]),
            'y': float(self.pos[i, 1]),
            'z': float(self.pos[i, 2]),
            'vx': float(self.vel[i, 0]),
            'vy': float(self.vel[i, 1]),
            'vz': float(self.vel[i, 2]),
            'qw': float(self.quat[i, 0]),
            'qx': float(self.quat[i, 1]),
            'qy': float(self.quat[i, 2]),
            'qz': float(self.quat[i, 3]),
            'roll': float(euler[0]),
            'pitch': float(euler[1]),
            'yaw': float(euler[2]),
            'motor_rpms': [float(self.motor_omega[i, m] * 60.0 / (2.0 * np.pi)) for m in range(4)],
            'battery': float(self.battery[i]),
        }

        # LIDAR
        lidar = [float(self.lidar_distances[i, r]) for r in range(12)]

        # Projectiles
        projectiles = []
        for j in range(MAX_PROJECTILES):
            if self.proj_active[i, j]:
                projectiles.append({
                    'x': float(self.proj_pos[i, j, 0]),
                    'y': float(self.proj_pos[i, j, 1]),
                    'z': float(self.proj_pos[i, j, 2]),
                    'vx': float(self.proj_vel[i, j, 0]),
                    'vy': float(self.proj_vel[i, j, 1]),
                    'vz': float(self.proj_vel[i, j, 2]),
                })

        # Turret state: aim direction + muzzle flash
        turret_states = []
        for t in self.turrets:
            aim = self.pos[i] - t['pos']
            aim_len = np.linalg.norm(aim)
            if aim_len > 0.1:
                aim = aim / aim_len
            tid = t['id']
            turret_states.append({
                'id': tid,
                'ax': float(aim[0]), 'ay': float(aim[1]), 'az': float(aim[2]),
                'flash': tid in self._turret_flash,
            })

        # Wind info
        wind = {
            'base': [float(self.base_wind[i, k]) for k in range(3)],
            'turb': [float(self.turb_state[i, k]) for k in range(3)],
            'gust_active': bool(self.gust_remaining[i] > 0),
        }

        # Waypoint info
        gi = int(self.current_gate[i])
        waypoint = None
        if gi < self.total_gates:
            g = self.gates[gi]
            waypoint = {
                'x': float(g['center'][0]),
                'y': float(g['center'][1]),
                'z': float(g['center'][2]),
                'width': float(g['width']),
                'height': float(g['height']),
                'index': gi,
            }

        # Task-specific target info for non-gate completion modes
        if self._endless_mode:
            profile = {'name': f'Endless #{self._endless_scenario}', 'completion_mode': 'gates'}
            completion_mode = 'gates'
        else:
            profile = CURRICULUM_PROFILES[self.curriculum_level]
            completion_mode = profile.get('completion_mode', 'gates')
        task_target = None
        if completion_mode == 'altitude_change':
            cyl_idx = int(self.alt_current_cyl[i])
            task_target = {
                'type': 'altitude_change',
                'cylinders': [
                    {'x': float(self.alt_cylinders[i, c, 0]),
                     'y': float(self.alt_cylinders[i, c, 1]),
                     'z': float(self.alt_cylinders[i, c, 2]),
                     'radius': ALT_CYL_RADIUS,
                     'height': ALT_CYL_HEIGHT}
                    for c in range(3)
                ],
                'current_cyl': cyl_idx,
                'changes_done': int(self.alt_changes_done[i]),
                'changes_total': 3,
                'hold_time': float(self.alt_hold_time[i]),
                'hold_required': ALT_CHANGE_HOLD,
            }
        elif completion_mode == 'yaw':
            task_target = {
                'type': 'yaw',
                'target_yaw': float(self.target_yaw[i]),
                'changes_done': int(self.yaw_changes_done[i]),
                'changes_total': 3,
                'hold_time': float(self.yaw_hold_time[i]),
                'hold_required': YAW_TARGET_HOLD,
                'cx': float(self.start_pos[i, 0]),
                'cy': float(self.start_pos[i, 1]),
                'tolerance': float(YAW_TOLERANCE),
                'zone_radius': YAW_ZONE_RADIUS,
                'zone_alt_lo': ALTITUDE_TARGET_LO,
                'zone_alt_hi': ALTITUDE_TARGET_HI,
            }
        elif completion_mode == 'fly_to_point':
            task_target = {
                'type': 'fly_to_point',
                'x': float(self.fly_to_target[i, 0]),
                'y': float(self.fly_to_target[i, 1]),
                'z': float(self.fly_to_target[i, 2]),
                'hold_time': float(self.fly_to_hold_time[i]),
                'hold_required': FLY_TO_HOLD,
                'radius': FLY_TO_RADIUS,
            }
        elif completion_mode == 'hover':
            stage = int(self.hover_shrink_stage[i])
            shrink = HOVER_SHRINK_FACTOR ** stage
            alt_mid = (ALTITUDE_TARGET_LO + ALTITUDE_TARGET_HI) / 2.0
            alt_half = (ALTITUDE_TARGET_HI - ALTITUDE_TARGET_LO) / 2.0 * shrink
            task_target = {
                'type': 'hover',
                'hold_time': float(self.hover_duration[i]),
                'hold_required': 10.0 if stage == self._hover_stages - 1 else HOVER_TARGET_DURATION,
                'stage': stage,
                'max_stages': self._hover_stages,
                # Zone geometry — shrinks each stage
                'cx': float(self.start_pos[i, 0]),
                'cy': float(self.start_pos[i, 1]),
                'radius': float(HOVER_POS_RADIUS * shrink),
                'alt_lo': float(alt_mid - alt_half),
                'alt_hi': float(alt_mid + alt_half),
            }
        elif completion_mode == 'takeoff':
            task_target = {
                'type': 'takeoff',
                'target_alt': TAKEOFF_TARGET_ALT,
                'hold_time': float(self.hover_duration[i]),
                'hold_required': TAKEOFF_HOLD_DURATION,
                'cx': float(self.start_pos[i, 0]),
                'cy': float(self.start_pos[i, 1]),
                'radius': HOVER_POS_RADIUS * 2.0,
                'alt_lo': TAKEOFF_TARGET_ALT,
                'alt_hi': ALTITUDE_TARGET_HI,
            }
        elif completion_mode == 'land':
            task_target = {
                'type': 'land',
                'cx': float(self.start_pos[i, 0]),
                'cy': float(self.start_pos[i, 1]),
                'radius': HOVER_POS_RADIUS * 2.0,
            }

        state = {
            'scenario': 'drone',
            'drone': drone,
            'lidar': lidar,
            'projectiles': projectiles,
            'turrets': turret_states,
            'wind': wind,
            'waypoint': waypoint,
            'gates_passed': gi,
            'total_gates': self.total_gates,
            'step': int(self.step_count[i]),
            'max_steps': self.max_steps,
            'curriculum_level': self.curriculum_level,
            'level_name': profile.get('name', ''),
            'completion_mode': completion_mode,
            'task_target': task_target,
            'gps_denied': bool(self.gps_denied[i]),
            'jamming': float(self.jamming_intensity[i]),
            'ew_active': profile.get('ew', False) if not self._endless_mode else len(self.gps_denial_zones) > 0,
        }

        # Endless mode info
        if self._endless_mode:
            state['endless'] = {
                'scenario': self._endless_scenario,
                'difficulty': round(self._endless_difficulty, 2),
                'streak': self._endless_current_streak,
                'best_streak': self._endless_best_streak,
            }

        # User weapon stats
        if self._weapon_mode:
            state['weapon'] = {
                'shots': self._user_shots,
                'hits': self._user_hits,
                'accuracy': round(self._user_hits / max(1, self._user_shots) * 100, 1),
            }

        # Adversary drone
        if self._adversary_enabled and self.adv_active:
            state['adversary'] = {
                'active': True,
                'x': float(self.adv_pos[0]),
                'y': float(self.adv_pos[1]),
                'z': float(self.adv_pos[2]),
                'vx': float(self.adv_vel[0]),
                'vy': float(self.adv_vel[1]),
                'vz': float(self.adv_vel[2]),
            }

        return state

    def get_swarm_state(self) -> list:
        """Minimal state for all envs: [x, y, z, qw, qx, qy, qz] per drone."""
        result = []
        for i in range(self.n_envs):
            result.append([
                float(self.pos[i, 0]), float(self.pos[i, 1]), float(self.pos[i, 2]),
                float(self.quat[i, 0]), float(self.quat[i, 1]),
                float(self.quat[i, 2]), float(self.quat[i, 3]),
            ])
        return result

    # ══════════════════════════════════════════════════════════
    #  Curriculum Management
    # ══════════════════════════════════════════════════════════

    def check_curriculum_advance(self):
        """Check if we should advance to the next difficulty level.

        For endless mode: each completion immediately generates the next scenario
        (no 70% threshold — each is a unique challenge).
        For normal curriculum: advance when 70% completion rate over window.
        When level 15 reaches 70%, auto-transition to endless mode.
        """
        if self._endless_mode:
            # In endless mode, advancement is handled per-completion in step()
            return False

        if len(self.recent_completions) < 50:
            return False
        completion_rate = sum(self.recent_completions[-self.curriculum_window:]) / min(
            len(self.recent_completions), self.curriculum_window
        )
        if completion_rate >= 0.7:
            if self.curriculum_level < MAX_CURRICULUM_LEVEL:
                self.curriculum_level += 1
                self.course_data = _generate_course(self.curriculum_level, self.rng)
                self.obstacles = self.course_data['obstacles']
                self.gates = self.course_data['gates']
                self.total_gates = len(self.gates)
                self.turrets = self.course_data.get('turrets', [])
                self._course_changed = True
                self.recent_completions.clear()
                self._apply_curriculum_profile()
                self.reset()
                return True
            elif self.curriculum_level == MAX_CURRICULUM_LEVEL:
                # Level 15 mastered — auto-transition to endless mode
                self.set_endless_mode(True)
                return True
        return False

    def set_curriculum_level(self, level: int):
        """Manually set curriculum level (1-MAX_CURRICULUM_LEVEL)."""
        level = max(1, min(MAX_CURRICULUM_LEVEL, level))
        # Disable endless mode when manually setting a curriculum level
        if self._endless_mode:
            self._endless_mode = False
        if level != self.curriculum_level:
            self.curriculum_level = level
            self.course_data = _generate_course(level, self.rng)
            self.obstacles = self.course_data['obstacles']
            self.gates = self.course_data['gates']
            self.total_gates = len(self.gates)
            self.turrets = self.course_data.get('turrets', [])
            self.recent_completions.clear()
            self._apply_curriculum_profile()
            self.reset()

    # ══════════════════════════════════════════════════════════
    #  User Kinetic Weapon
    # ══════════════════════════════════════════════════════════

    def inject_user_projectile(self, target_x: float, target_y: float, target_z: float):
        """Fire a user-aimed projectile at the given world-space target.

        Projectile spawns from a fixed turret position at the arena edge.
        Uses simple lead targeting based on drone velocity.
        Reuses existing projectile arrays for physics and rendering.
        """
        self._user_shots += 1
        i = 0  # only env 0 (display env)

        # Turret position: arena edge, center Y, 10m altitude
        turret_pos = np.array([0.0, COURSE_D / 2.0, 10.0], dtype=np.float32)
        target = np.array([target_x, target_y, target_z], dtype=np.float32)

        # Simple lead targeting: predict where drone will be
        to_target = target - turret_pos
        flight_time = np.linalg.norm(to_target) / 20.0  # user projectile speed = 20 m/s
        lead_target = target + self.vel[i] * flight_time * 0.3

        direction = lead_target - turret_pos
        dist = np.linalg.norm(direction)
        if dist < 0.1:
            return
        direction /= dist

        # Find an inactive projectile slot
        for j in range(MAX_PROJECTILES):
            if not self.proj_active[i, j]:
                self.proj_pos[i, j] = turret_pos
                self.proj_vel[i, j] = direction * 20.0  # 20 m/s (faster than AI at 15)
                self.proj_active[i, j] = True
                self.proj_life[i, j] = 0.0
                return

    # ══════════════════════════════════════════════════════════
    #  Adversarial Drone (Proportional Navigation)
    # ══════════════════════════════════════════════════════════

    def set_adversary(self, enabled: bool, lethal: bool = False):
        """Toggle adversary drone. Lethal = collision ends episode."""
        self._adversary_enabled = enabled
        self._adversary_lethal = lethal
        if not enabled:
            self.adv_active = False

    def _spawn_adversary(self):
        """Spawn adversary drone at a random arena edge, 25m from primary."""
        D = self._endless_difficulty if self._endless_mode else 3.0
        # Pick random edge
        edge = self.rng.integers(0, 4)
        if edge == 0:    # left
            ax, ay = 1.0, self.rng.uniform(5.0, COURSE_D - 5.0)
        elif edge == 1:  # right
            ax, ay = COURSE_W - 1.0, self.rng.uniform(5.0, COURSE_D - 5.0)
        elif edge == 2:  # bottom
            ax, ay = self.rng.uniform(5.0, COURSE_W - 5.0), 1.0
        else:            # top
            ax, ay = self.rng.uniform(5.0, COURSE_W - 5.0), COURSE_D - 1.0
        az = self.rng.uniform(3.0, 10.0)
        self.adv_pos = np.array([ax, ay, az], dtype=np.float32)
        self.adv_vel = np.zeros(3, dtype=np.float32)
        self.adv_active = True
        self.adv_lock_lost_timer = 0.0

    def _update_adversary(self, dt: float):
        """Update adversary drone using 3D proportional navigation.

        PN guidance: a_cmd = N * V_c * dLOS/dt
        N=3 is the standard pursuit constant. Produces realistic intercept curves.
        """
        if not self.adv_active:
            return

        D = self._endless_difficulty if self._endless_mode else 3.0
        max_speed = 6.0 + D * 0.8    # 6-10 m/s
        accel = 4.0                    # m/s²
        turn_rate = 2.0                # rad/s

        drone_pos = self.pos[0]
        drone_vel = self.vel[0]

        # Periodic "lock loss" — straight-line flight
        lock_loss_interval = max(5.0, 12.0 - D * 1.5)
        self.adv_lock_lost_timer += dt
        lock_lost = (self.adv_lock_lost_timer % lock_loss_interval) < 2.0

        if lock_lost:
            # Fly straight (maintain current velocity direction)
            speed = np.linalg.norm(self.adv_vel)
            if speed > 0.1:
                self.adv_pos += self.adv_vel * dt
            return

        # ── Proportional Navigation ─────────────────────────
        # LOS (line of sight) vector
        los = drone_pos - self.adv_pos
        los_dist = np.linalg.norm(los)
        if los_dist < 0.1:
            return

        los_unit = los / los_dist

        # Closing velocity
        rel_vel = drone_vel - self.adv_vel
        v_closing = -np.dot(rel_vel, los_unit)

        # LOS rate: omega = (V_rel x LOS) / |LOS|^2
        los_rate = np.cross(rel_vel, los_unit) / (los_dist + 0.1)

        # PN acceleration command: a = N * Vc * omega_LOS
        N = 3.0  # navigation constant
        a_cmd = N * max(v_closing, 1.0) * los_rate

        # Clamp acceleration
        a_mag = np.linalg.norm(a_cmd)
        if a_mag > accel:
            a_cmd = a_cmd / a_mag * accel

        # Add forward acceleration toward target
        desired_dir = los_unit
        current_speed = np.linalg.norm(self.adv_vel)
        if current_speed < max_speed:
            a_cmd += desired_dir * min(accel, max_speed - current_speed) / dt * 0.1

        # Update velocity with turn rate limit
        new_vel = self.adv_vel + a_cmd * dt
        new_speed = np.linalg.norm(new_vel)
        if new_speed > max_speed:
            new_vel = new_vel / new_speed * max_speed

        # Limit turn rate (angle change per second)
        if current_speed > 0.5 and new_speed > 0.5:
            old_dir = self.adv_vel / current_speed
            new_dir = new_vel / new_speed
            cos_angle = np.clip(np.dot(old_dir, new_dir), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            max_angle = turn_rate * dt
            if angle > max_angle and angle > 0.001:
                # Slerp between old and new direction
                t = max_angle / angle
                blended_dir = old_dir * (1.0 - t) + new_dir * t
                blended_dir /= np.linalg.norm(blended_dir) + 1e-8
                new_vel = blended_dir * new_speed

        self.adv_vel = new_vel
        self.adv_pos += self.adv_vel * dt

        # Clamp to arena bounds
        self.adv_pos[0] = np.clip(self.adv_pos[0], 0.5, COURSE_W - 0.5)
        self.adv_pos[1] = np.clip(self.adv_pos[1], 0.5, COURSE_D - 0.5)
        self.adv_pos[2] = np.clip(self.adv_pos[2], 1.0, COURSE_H - 1.0)

    def _check_adversary_collision(self) -> bool:
        """Check if adversary has intercepted the primary drone (env 0).
        Returns True if collision detected."""
        if not self.adv_active:
            return False
        dist = np.linalg.norm(self.adv_pos - self.pos[0])
        intercept_dist = DRONE_RADIUS + 0.5
        return dist < intercept_dist
