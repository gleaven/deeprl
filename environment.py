"""Vectorized Soccer Environment — pure NumPy, supports 1v1 and 2v2.

Action space: [move_x, move_y, kick_power, kick_dir_x, kick_dir_y, kick_loft, jump]
  - move_x, move_y: movement force direction [-1, 1]
  - kick_power: kick activation + power [-1, 1], >threshold triggers kick
  - kick_dir_x, kick_dir_y: offset to kick direction [-1, 1], enables aimed kicks
  - kick_loft: vertical kick angle [-1, 1], mapped to [0, 0.7] loft fraction
  - jump: vertical jump [-1, 1], >0.3 triggers jump with variable height
"""

import numpy as np

# ── Field Constants ────────────────────────────────────────────
FIELD_W = 30.0          # meters — proportional to player height (~29 player-heights)
FIELD_H = 20.0          # wider for realistic 1v1 futsal proportions
GOAL_W = 5.0            # goal opening width (centered on end lines)
GOAL_Y_MIN = (FIELD_H - GOAL_W) / 2.0
GOAL_Y_MAX = (FIELD_H + GOAL_W) / 2.0

# ── Physics Constants ──────────────────────────────────────────
AGENT_RADIUS = 0.5
BALL_RADIUS = 0.18       # 0.36m diameter — proportional to players (was 0.35)
AGENT_MASS = 1.0
BALL_MASS = 0.2          # lighter ball, more responsive to kicks (was 0.3)
AGENT_FRICTION = 0.85    # velocity multiplier per step
BALL_FRICTION = 0.95
MAX_AGENT_SPEED = 8.0    # m/s
MAX_BALL_SPEED = 15.0
KICK_RANGE = 1.2         # tighter foot contact distance (was 1.5)
KICK_THRESHOLD = 0.3    # lowered from 0.5 — allows soft dribble touches
KICK_FORCE = 12.0       # max impulse magnitude applied to ball
AGENT_FORCE = 10.0      # max force agents can apply to themselves per step
DT = 0.05               # physics timestep (seconds)

# ── 3D Ball Physics ───────────────────────────────────────────
GRAVITY = 9.8            # m/s² downward
BOUNCE_COEFF = 0.6       # velocity retained on ground bounce
AIR_FRICTION = 0.995     # less drag than ground rolling (0.95)
GOAL_HEIGHT = 1.5        # matches visual goal height — ball must be below this to score

# ── Jump Constants ────────────────────────────────────────────
JUMP_IMPULSE = 4.5       # m/s vertical velocity on jump
JUMP_THRESHOLD = 0.7     # high threshold — requires deliberate action (was 0.3)
JUMP_COOLDOWN_STEPS = 45 # ~2.25s real time between jumps (was 30)
AGENT_REACH = 1.2        # vertical reach of agent body (feet to head)
AIR_CONTROL = 0.15       # heavily reduced XY force while airborne — committed action
AIRBORNE_PENALTY = 0.02  # per-step cost while in the air — discourages prolonged jumping

# ── Action / Observation sizes ─────────────────────────────────
ACTION_SIZE = 7          # [move_x, move_y, kick_power, kick_dir_x, kick_dir_y, kick_loft, jump]

# ── Episode Constants ──────────────────────────────────────────
MAX_STEPS = 500
MAX_GOALS = 1           # first to N goals ends episode (sudden death default)

# ── Dribble / Skill Constants ─────────────────────────────────
DRIBBLE_RANGE = 1.2     # close ball control range (tighter than KICK_RANGE)
JUKE_ANGLE_THRESHOLD = 0.5  # cosine similarity threshold for direction change

# ── Reward Constants ───────────────────────────────────────────
# Tuned for fast learning progression: approach → kick → direct → score
# Approximate per-episode totals in parentheses:
GOAL_REWARD = 10.0           # dominant signal (~10 per goal scored)
APPROACH_BALL_REWARD = 0.02  # move toward ball (~4/ep)
BALL_TOWARD_GOAL_REWARD = 0.05  # kick ball goalward (~3.5/ep) — 2.5× approach
KICK_BALL_REWARD = 0.1       # contact the ball (~3/ep)
DRAW_PENALTY = 3.0           # 30% of a goal — strong push to score
ENERGY_PENALTY = 0.003       # mild anti-flailing (~1.5/ep)
WALL_PENALTY = 0.02          # mild wall avoidance (~1/ep)
CORNER_BALL_PENALTY = 0.03   # discourages ball stuck in corners
BALL_WALL_PENALTY = 0.05     # discourages ball near any wall — keeps play central
BALL_WALL_MARGIN = 2.5       # meters from wall edge (scaled for 30×20 field)

# Out-of-bounds penalties/rewards
OUT_OF_BOUNDS_PENALTY = 0.5   # penalty for kicking ball out of bounds
THROW_IN_REWARD = 0.25        # reward for team receiving the throw-in (half the penalty)
THROW_IN_SPEED = 5.0          # ball velocity on auto throw-in (moderate toss)

# Skill rewards — encourage dribbling, ball control, and juking
DRIBBLE_REWARD = 0.08        # moving with ball under close control
POSSESSION_REWARD = 0.01     # having ball nearby (within 2m)
JUKE_REWARD = 0.12           # sharp direction change while possessing ball near opponent
JUMP_PENALTY = 0.3           # cost per jump — heavy penalty makes jumping rare and tactical

# Corner detection
CORNER_RADIUS = 4.5          # scaled for 30×20 field
_CORNERS = np.array([
    [0.0, 0.0], [FIELD_W, 0.0],
    [0.0, FIELD_H], [FIELD_W, FIELD_H],
], dtype=np.float32)

# ── Observation size ───────────────────────────────────────────
# self_pos(2) + self_vel(2) + teammate_rel_pos(2) + teammate_rel_vel(2)
# + opp1_rel_pos(2) + opp1_rel_vel(2) + opp2_rel_pos(2) + opp2_rel_vel(2)
# + ball_rel_pos(2) + ball_rel_vel(2) + own_goal_rel(2) + opp_goal_rel(2)
# + score_diff(1) + step_frac(1) + team_flag(1) + closest_to_ball(1)
# + has_possession(1) + ball_speed(1) + ball_z(1) + ball_vz(1)
# + self_z(1) + self_vz(1)
OBS_SIZE = 34

# Off-field parking position for inactive agents
_PARK_POS = np.array([-100.0, -100.0], dtype=np.float32)


class VectorizedSoccerEnv:
    """N parallel soccer matches as batched NumPy arrays. Supports 1v1 and 2v2."""

    def __init__(self, n_envs: int = 64, players_per_team: int = 1, seed: int = 42):
        self.n_envs = n_envs
        self.players_per_team = players_per_team  # 1 or 2
        self.rng = np.random.default_rng(seed)

        # State arrays — always [N, 4, 2] for max 2v2
        self.agent_pos = np.zeros((n_envs, 4, 2), dtype=np.float32)
        self.agent_vel = np.zeros((n_envs, 4, 2), dtype=np.float32)
        self.ball_pos = np.zeros((n_envs, 2), dtype=np.float32)
        self.ball_vel = np.zeros((n_envs, 2), dtype=np.float32)
        self.ball_z = np.zeros(n_envs, dtype=np.float32)   # height above ground
        self.ball_vz = np.zeros(n_envs, dtype=np.float32)  # vertical velocity
        self.score = np.zeros((n_envs, 2), dtype=np.int32)
        self.step_count = np.zeros(n_envs, dtype=np.int32)
        self.done = np.zeros(n_envs, dtype=bool)

        # Previous ball distances for reward shaping
        self._prev_ball_dist = np.zeros((n_envs, 4), dtype=np.float32)
        self._prev_ball_goal_dist = np.zeros(n_envs, dtype=np.float32)

        # Goal positions
        self.left_goal = np.array([0.0, FIELD_H / 2.0], dtype=np.float32)
        self.right_goal = np.array([FIELD_W, FIELD_H / 2.0], dtype=np.float32)

        # ── Leg / skill state (for rendering + reward) ─────────
        self.agent_facing = np.zeros((n_envs, 4), dtype=np.float32)  # radians
        self.agent_kick_state = np.zeros((n_envs, 4), dtype=np.float32)  # 0-1, decays
        self.agent_kick_power = np.zeros((n_envs, 4), dtype=np.float32)
        self.agent_dribbling = np.zeros((n_envs, 4), dtype=bool)
        self.agent_juking = np.zeros((n_envs, 4), dtype=bool)
        self._kick_leg = np.zeros((n_envs, 4), dtype=np.int32)  # 0=left, 1=right
        self._prev_agent_vel = np.zeros((n_envs, 4, 2), dtype=np.float32)
        self.agent_contact_type = np.zeros((n_envs, 4), dtype=np.int32)  # 0=none,1=foot,2=knee,3=header
        self.agent_z = np.zeros((n_envs, 4), dtype=np.float32)    # height above ground
        self.agent_vz = np.zeros((n_envs, 4), dtype=np.float32)   # vertical velocity
        self.agent_jump_cooldown = np.zeros((n_envs, 4), dtype=np.int32)
        self._last_kicker_team = np.full(n_envs, -1, dtype=np.int32)  # -1=nobody, 0/1=team

        # Configurable reward weights (all adjustable via UI sliders)
        self.goal_reward = GOAL_REWARD
        self.approach_reward = APPROACH_BALL_REWARD
        self.ball_goal_reward = BALL_TOWARD_GOAL_REWARD
        self.kick_reward = KICK_BALL_REWARD
        self.draw_penalty = DRAW_PENALTY
        self.energy_penalty = ENERGY_PENALTY
        self.wall_penalty = WALL_PENALTY
        self.corner_ball_penalty = CORNER_BALL_PENALTY
        self.ball_wall_penalty = BALL_WALL_PENALTY

        # Skill reward weights
        self.dribble_reward = DRIBBLE_REWARD
        self.possession_reward = POSSESSION_REWARD
        self.juke_reward = JUKE_REWARD

        # Configurable episode settings
        self.max_steps = MAX_STEPS
        self.max_goals = MAX_GOALS

        self.reset()

    @property
    def active_indices(self) -> list[int]:
        """Indices of active agents. 1v1=[0,2], 2v2=[0,1,2,3]."""
        if self.players_per_team == 1:
            return [0, 2]
        return [0, 1, 2, 3]

    @property
    def n_active(self) -> int:
        return self.players_per_team * 2

    def set_players_per_team(self, n: int):
        """Switch between 1v1 and 2v2. Resets all envs."""
        assert n in (1, 2), "players_per_team must be 1 or 2"
        self.players_per_team = n
        self.reset()

    def reset(self, env_mask: np.ndarray | None = None):
        """Reset all envs or only those indicated by env_mask."""
        if env_mask is None:
            env_mask = np.ones(self.n_envs, dtype=bool)

        n_reset = env_mask.sum()
        if n_reset == 0:
            return self.get_obs()

        cx, cy = FIELD_W / 2.0, FIELD_H / 2.0

        # Fixed kickoff positions — mirrored, equally distant from ball
        # 1v1: one per side facing the ball
        self.agent_pos[env_mask, 0] = np.array([cx - 5.0, cy], dtype=np.float32)
        self.agent_pos[env_mask, 2] = np.array([cx + 5.0, cy], dtype=np.float32)

        if self.players_per_team == 2:
            self.agent_pos[env_mask, 1] = np.array([cx - 4.0, cy + 4.0], dtype=np.float32)
            self.agent_pos[env_mask, 3] = np.array([cx + 4.0, cy - 4.0], dtype=np.float32)
        else:
            # Park inactive agents off-field
            self.agent_pos[env_mask, 1] = _PARK_POS
            self.agent_pos[env_mask, 3] = _PARK_POS

        # All velocities zero at kickoff
        self.agent_vel[env_mask] = 0.0

        # Ball at center, stationary, on ground
        self.ball_pos[env_mask] = np.array([cx, cy], dtype=np.float32)
        self.ball_vel[env_mask] = 0.0
        self.ball_z[env_mask] = 0.0
        self.ball_vz[env_mask] = 0.0

        self.score[env_mask] = 0
        self.step_count[env_mask] = 0
        self.done[env_mask] = False

        # Reset leg / skill state
        self.agent_facing[env_mask] = 0.0
        self.agent_kick_state[env_mask] = 0.0
        self.agent_kick_power[env_mask] = 0.0
        self.agent_dribbling[env_mask] = False
        self.agent_juking[env_mask] = False
        self._kick_leg[env_mask] = 0
        self._prev_agent_vel[env_mask] = 0.0
        self.agent_contact_type[env_mask] = 0
        self.agent_z[env_mask] = 0.0
        self.agent_vz[env_mask] = 0.0
        self.agent_jump_cooldown[env_mask] = 0
        self._last_kicker_team[env_mask] = -1

        # Init reward shaping baselines
        self._update_distances()

        return self.get_obs()

    def _rand_pos(self, n: int, xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
        x = self.rng.uniform(xmin, xmax, n).astype(np.float32)
        y = self.rng.uniform(ymin, ymax, n).astype(np.float32)
        return np.stack([x, y], axis=-1)

    def _update_distances(self):
        """Update distance caches for reward shaping."""
        ball_exp = self.ball_pos[:, np.newaxis, :]  # [N, 1, 2]
        diff = self.agent_pos - ball_exp             # [N, 4, 2]
        self._prev_ball_dist = np.linalg.norm(diff, axis=-1)  # [N, 4]

        self._prev_ball_goal_dist = np.linalg.norm(
            self.ball_pos - self.right_goal[np.newaxis, :], axis=-1
        )

    def step(self, actions: np.ndarray):
        """
        Step all environments.

        Args:
            actions: [N, n_active, ACTION_SIZE] — per active agent:
                     [move_x, move_y, kick_power, kick_dir_x, kick_dir_y] in [-1, 1]

        Returns:
            obs: [N, n_active, OBS_SIZE]
            rewards: [N, n_active]
            dones: [N]
            infos: dict with episode stats
        """
        active = self.active_indices

        # Expand actions into full 4-agent array
        full_actions = np.zeros((self.n_envs, 4, ACTION_SIZE), dtype=np.float32)
        for i, idx in enumerate(active):
            full_actions[:, idx] = actions[:, i]
        full_actions = np.clip(full_actions, -1.0, 1.0)

        move = full_actions[:, :, :2]           # [N, 4, 2]
        kick_power = full_actions[:, :, 2]      # [N, 4]  — activation + power
        kick_dir_input = full_actions[:, :, 3:5] # [N, 4, 2] — kick direction offset
        kick_loft_input = full_actions[:, :, 5]  # [N, 4] — vertical loft angle
        jump_input = full_actions[:, :, 6]       # [N, 4] — jump trigger

        # ── 0. Jump mechanic ──────────────────────────────────────
        self.agent_jump_cooldown = np.maximum(0, self.agent_jump_cooldown - 1)
        jumped_this_step = np.zeros((self.n_envs, 4), dtype=bool)
        for idx in active:
            want_jump = jump_input[:, idx] > JUMP_THRESHOLD
            on_ground = self.agent_z[:, idx] < 0.01
            off_cooldown = self.agent_jump_cooldown[:, idx] == 0
            do_jump = want_jump & on_ground & off_cooldown
            if do_jump.any():
                # Variable jump height: 50% to 100% of JUMP_IMPULSE
                power = np.clip(jump_input[:, idx], 0.3, 1.0)
                self.agent_vz[do_jump, idx] = JUMP_IMPULSE * (0.5 + 0.5 * power[do_jump])
                self.agent_jump_cooldown[do_jump, idx] = JUMP_COOLDOWN_STEPS
                jumped_this_step[do_jump, idx] = True

        # Agent vertical physics: gravity → integrate → land
        for idx in active:
            self.agent_vz[:, idx] -= GRAVITY * DT
            self.agent_z[:, idx] += self.agent_vz[:, idx] * DT
            landed = self.agent_z[:, idx] <= 0
            self.agent_z[landed, idx] = 0
            self.agent_vz[landed, idx] = 0

        # ── 1. Apply agent forces (only active agents) ────────────
        force = move * AGENT_FORCE
        for idx in active:
            # Reduced control while airborne
            airborne = self.agent_z[:, idx] > 0.01
            force_scale = np.where(airborne, AIR_CONTROL, 1.0)
            self.agent_vel[:, idx] += force[:, idx] * force_scale[:, np.newaxis] * DT / AGENT_MASS
            self.agent_vel[:, idx] *= AGENT_FRICTION

        # Clamp active agent speed
        for idx in active:
            speed = np.linalg.norm(self.agent_vel[:, idx], axis=-1, keepdims=True)
            too_fast = speed > MAX_AGENT_SPEED
            scale = np.where(too_fast, MAX_AGENT_SPEED / (speed + 1e-8), 1.0)
            self.agent_vel[:, idx] *= scale

        # Update active positions
        for idx in active:
            self.agent_pos[:, idx] += self.agent_vel[:, idx] * DT

        # ── 2. Kick mechanics (variable power + directional + loft) ─
        ball_exp = self.ball_pos[:, np.newaxis, :]  # [N, 1, 2]
        to_ball = ball_exp - self.agent_pos          # [N, 4, 2]
        dist_to_ball = np.linalg.norm(to_ball, axis=-1)  # [N, 4]

        can_kick = (dist_to_ball < KICK_RANGE) & (kick_power > KICK_THRESHOLD)  # [N, 4]
        # Mask out inactive agents
        for idx in set(range(4)) - set(active):
            can_kick[:, idx] = False

        # ── Height-aware contact types (relative to agent height) ─
        # rel_z = ball height relative to agent: 0 = at feet, 1.0+ = above head
        ball_z_exp = self.ball_z[:, np.newaxis]   # [N, 1]
        rel_z = ball_z_exp - self.agent_z         # [N, 4]

        # Ball must be within vertical reach
        ball_reachable = (rel_z >= -0.1) & (rel_z < AGENT_REACH)
        can_kick &= ball_reachable

        # Contact type based on relative height
        contact_type = np.zeros((self.n_envs, 4), dtype=np.int32)
        contact_type[can_kick & (rel_z < 0.4)] = 1                        # foot
        contact_type[can_kick & (rel_z >= 0.3) & (rel_z < 0.8)] = 2       # knee/shin
        contact_type[can_kick & (rel_z >= 0.7)] = 3                        # header
        # If no contact assigned but can_kick, default to foot
        contact_type[can_kick & (contact_type == 0)] = 1
        self.agent_contact_type = contact_type

        # Default kick direction: toward ball
        default_dir = to_ball / (dist_to_ball[:, :, np.newaxis] + 1e-8)  # [N, 4, 2]

        # Blend in directional control — agent can aim kicks
        combined_dir = default_dir + kick_dir_input * 0.5
        combined_norm = np.linalg.norm(combined_dir, axis=-1, keepdims=True) + 1e-8
        kick_direction = combined_dir / combined_norm  # [N, 4, 2]

        # Variable force: scales with kick_power (20% at threshold, 100% at max)
        power_frac = np.clip(
            (kick_power - KICK_THRESHOLD) / (1.0 - KICK_THRESHOLD), 0.0, 1.0
        )
        actual_force = KICK_FORCE * (0.2 + 0.8 * power_frac)  # [N, 4]

        # Reduce power for knee/shin contact (-20%)
        knee_mask = contact_type == 2
        actual_force[knee_mask] *= 0.8

        # Headers: force in agent movement direction (not toward ball), no loft
        for idx in active:
            is_header = contact_type[:, idx] == 3
            if is_header.any():
                heading_dir = self.agent_vel[:, idx].copy()
                heading_spd = np.linalg.norm(heading_dir, axis=-1, keepdims=True) + 1e-8
                heading_dir /= heading_spd
                kick_direction[is_header, idx] = heading_dir[is_header]

        # ── Loft: vertical kick component ─────────────────────
        loft_frac = np.clip((kick_loft_input + 1.0) / 2.0, 0.0, 0.7)  # [-1,1] → [0, 0.7]
        # Headers don't loft (ball is already high)
        loft_frac[contact_type == 3] = 0.0

        # Horizontal force scaled down by loft
        horiz_scale = 1.0 - loft_frac * 0.5  # up to 35% reduction
        horiz_force = actual_force * horiz_scale

        kick_impulse = (
            kick_direction
            * can_kick[:, :, np.newaxis]
            * horiz_force[:, :, np.newaxis]
        )
        total_kick = kick_impulse.sum(axis=1)
        self.ball_vel += total_kick * DT / BALL_MASS

        # Vertical impulse from loft (3x boost — single-frame impulse must overcome gravity)
        vert_force = actual_force * loft_frac * can_kick.astype(np.float32) * 3.0
        total_vkick = vert_force.sum(axis=1)  # [N]
        self.ball_vz += total_vkick * DT / BALL_MASS

        # ── 2b. Update kick animation state + last kicker tracking ─
        self.agent_kick_state *= 0.7  # decay previous kicks
        for idx in active:
            just_kicked = can_kick[:, idx]
            if just_kicked.any():
                self.agent_kick_state[just_kicked, idx] = power_frac[just_kicked, idx]
                self.agent_kick_power[just_kicked, idx] = power_frac[just_kicked, idx]
                self._kick_leg[just_kicked, idx] = 1 - self._kick_leg[just_kicked, idx]
                team = 0 if idx < 2 else 1
                self._last_kicker_team[just_kicked] = team

        # ── 2c. Update facing direction ────────────────────────
        for idx in active:
            spd = np.linalg.norm(self.agent_vel[:, idx], axis=-1)
            moving = spd > 0.3
            if moving.any():
                self.agent_facing[moving, idx] = np.arctan2(
                    self.agent_vel[moving, idx, 1],
                    self.agent_vel[moving, idx, 0],
                )

        # ── 2d. Detect dribbling and juking ────────────────────
        agent_speed = np.linalg.norm(self.agent_vel, axis=-1)  # [N, 4]
        is_dribbling = (dist_to_ball < DRIBBLE_RANGE) & (agent_speed > 1.0)
        for idx in set(range(4)) - set(active):
            is_dribbling[:, idx] = False
        self.agent_dribbling = is_dribbling

        # Juke: velocity direction changed sharply while possessing ball near opponent
        curr_vel = self.agent_vel.copy()
        prev_vel = self._prev_agent_vel.copy()
        curr_spd = np.linalg.norm(curr_vel, axis=-1) + 1e-8   # [N, 4]
        prev_spd = np.linalg.norm(prev_vel, axis=-1) + 1e-8
        # Cosine similarity between current and previous velocity
        cos_sim = (curr_vel * prev_vel).sum(axis=-1) / (curr_spd * prev_spd)
        dir_changed = (cos_sim < JUKE_ANGLE_THRESHOLD) & (curr_spd > 1.5) & (prev_spd > 1.5)

        near_ball = dist_to_ball < 2.0
        # Check opponent proximity (any opponent within 3m)
        near_opp = np.zeros((self.n_envs, 4), dtype=bool)
        for idx in active:
            team = 0 if idx < 2 else 1
            for opp_idx in active:
                if (0 if opp_idx < 2 else 1) != team:
                    d = np.linalg.norm(
                        self.agent_pos[:, idx] - self.agent_pos[:, opp_idx], axis=-1
                    )
                    near_opp[:, idx] |= (d < 3.0)
        is_juking = dir_changed & near_ball & near_opp
        for idx in set(range(4)) - set(active):
            is_juking[:, idx] = False
        self.agent_juking = is_juking

        # Store velocity for next step's juke detection
        self._prev_agent_vel = self.agent_vel.copy()

        # ── 3. Ball physics (2D + 3D vertical) ────────────────
        airborne = self.ball_z > 0.01
        # Ground friction for grounded balls, air friction for airborne
        friction = np.where(airborne, AIR_FRICTION, BALL_FRICTION)
        self.ball_vel *= friction[:, np.newaxis]

        ball_speed = np.linalg.norm(self.ball_vel, axis=-1, keepdims=True)
        too_fast_ball = ball_speed > MAX_BALL_SPEED
        self.ball_vel *= np.where(too_fast_ball, MAX_BALL_SPEED / (ball_speed + 1e-8), 1.0)
        self.ball_pos += self.ball_vel * DT

        # Vertical physics: gravity → integrate → bounce
        self.ball_vz -= GRAVITY * DT
        self.ball_z += self.ball_vz * DT
        on_ground = self.ball_z <= 0
        self.ball_z[on_ground] = 0
        self.ball_vz[on_ground] = -self.ball_vz[on_ground] * BOUNCE_COEFF
        # Kill tiny bounces
        tiny_bounce = on_ground & (np.abs(self.ball_vz) < 0.3)
        self.ball_vz[tiny_bounce] = 0

        # ── 4. Collisions ─────────────────────────────────────
        self._resolve_agent_agent_collisions()
        self._resolve_agent_ball_collisions()

        # ── 5. Boundaries ─────────────────────────────────────
        wall_clip_agents = self._clamp_to_field_agents()
        ball_out = self._clamp_to_field_ball()

        # ── 6. Goal detection ──────────────────────────────────
        goal_team0, goal_team1 = self._check_goals()

        # ── 7. Update scores ───────────────────────────────────
        self.score[:, 0] += goal_team0.astype(np.int32)
        self.score[:, 1] += goal_team1.astype(np.int32)

        scored = goal_team0 | goal_team1
        if scored.any():
            self.ball_pos[scored] = np.array([FIELD_W / 2.0, FIELD_H / 2.0], dtype=np.float32)
            self.ball_vel[scored] = self.rng.uniform(-0.3, 0.3, (scored.sum(), 2)).astype(np.float32)
            self.ball_z[scored] = 0.0
            self.ball_vz[scored] = 0.0

        # ── 8. Compute rewards ─────────────────────────────────
        rewards_full = self._compute_rewards(move, goal_team0, goal_team1, wall_clip_agents, can_kick)

        # ── 8b. Skill rewards (dribble, possession, juke) ──────
        rewards_full += self._compute_skill_rewards(dist_to_ball, is_dribbling, is_juking)

        # ── 8c. Instant throw-in: teleport players + auto-throw ───
        if ball_out.any():
            center = np.array([FIELD_W / 2.0, FIELD_H / 2.0])
            for env_i in np.where(ball_out)[0]:
                kicker_team = self._last_kicker_team[env_i]
                if kicker_team < 0:
                    continue  # no known kicker, skip teleport

                # Penalty/reward
                if kicker_team == 0:
                    rewards_full[env_i, 0] -= OUT_OF_BOUNDS_PENALTY
                    rewards_full[env_i, 1] -= OUT_OF_BOUNDS_PENALTY
                    rewards_full[env_i, 2] += THROW_IN_REWARD
                    rewards_full[env_i, 3] += THROW_IN_REWARD
                else:
                    rewards_full[env_i, 2] -= OUT_OF_BOUNDS_PENALTY
                    rewards_full[env_i, 3] -= OUT_OF_BOUNDS_PENALTY
                    rewards_full[env_i, 0] += THROW_IN_REWARD
                    rewards_full[env_i, 1] += THROW_IN_REWARD

                # (a) Teleport throw-in player to ball
                throw_team = 1 - kicker_team
                if self.players_per_team == 1:
                    throw_idx = 2 if throw_team == 1 else 0
                    def_idx = 0 if kicker_team == 0 else 2
                    throw_indices = [throw_idx]
                    def_indices = [def_idx]
                else:
                    throw_indices = [0, 1] if throw_team == 0 else [2, 3]
                    def_indices = [0, 1] if kicker_team == 0 else [2, 3]

                ball_xy = self.ball_pos[env_i].copy()
                # Place thrower just outside boundary near ball — runs in after throw
                throw_pos = ball_xy.copy()
                # Offset 1m outward from nearest boundary
                if ball_xy[1] <= BALL_RADIUS + 0.1:        # bottom boundary
                    throw_pos[1] = -1.0
                elif ball_xy[1] >= FIELD_H - BALL_RADIUS - 0.1:  # top boundary
                    throw_pos[1] = FIELD_H + 1.0
                elif ball_xy[0] <= BALL_RADIUS + 0.1:      # left boundary
                    throw_pos[0] = -1.0
                elif ball_xy[0] >= FIELD_W - BALL_RADIUS - 0.1:  # right boundary
                    throw_pos[0] = FIELD_W + 1.0
                self.agent_pos[env_i, throw_indices[0]] = throw_pos
                self.agent_vel[env_i, throw_indices[0]] = 0.0
                self.agent_z[env_i, throw_indices[0]] = 0.0
                self.agent_vz[env_i, throw_indices[0]] = 0.0

                # (b) Place defender midway between ball and own goal
                own_goal_x = 0.0 if kicker_team == 0 else FIELD_W
                def_x = (ball_xy[0] + own_goal_x) / 2.0
                def_y = np.clip(ball_xy[1], AGENT_RADIUS, FIELD_H - AGENT_RADIUS)
                self.agent_pos[env_i, def_indices[0]] = [def_x, def_y]
                self.agent_vel[env_i, def_indices[0]] = 0.0
                self.agent_z[env_i, def_indices[0]] = 0.0
                self.agent_vz[env_i, def_indices[0]] = 0.0

                # 2v2: second thrower near ball, second defender near goal
                if self.players_per_team == 2:
                    self.agent_pos[env_i, throw_indices[1]] = ball_xy + [0, 2.0]
                    self.agent_vel[env_i, throw_indices[1]] = 0.0
                    self.agent_z[env_i, throw_indices[1]] = 0.0
                    self.agent_vz[env_i, throw_indices[1]] = 0.0
                    self.agent_pos[env_i, def_indices[1]] = [own_goal_x + (1.0 if kicker_team == 0 else -1.0) * 3.0, FIELD_H / 2.0]
                    self.agent_vel[env_i, def_indices[1]] = 0.0
                    self.agent_z[env_i, def_indices[1]] = 0.0
                    self.agent_vz[env_i, def_indices[1]] = 0.0

                # (c) Agent-directed throw-in: use thrower's move direction
                # The agent learns WHERE to throw — avoids throwing at opponent
                throw_dir = move[env_i, throw_indices[0]].copy()  # [2]
                throw_mag = np.linalg.norm(throw_dir) + 1e-8
                if throw_mag > 0.1:
                    throw_dir /= throw_mag  # normalize to unit direction
                else:
                    # No clear direction — default inward toward center
                    throw_dir = center - ball_xy
                    throw_dir /= (np.linalg.norm(throw_dir) + 1e-8)
                self.ball_vel[env_i] = throw_dir.astype(np.float32) * THROW_IN_SPEED

        # ── 8d. Jump penalties — heavy cost makes jumping rare and tactical ──
        rewards_full -= jumped_this_step.astype(np.float32) * JUMP_PENALTY
        # Continuous airborne penalty — every step in the air costs
        airborne_agents = self.agent_z > 0.05  # [N, 4]
        rewards_full -= airborne_agents.astype(np.float32) * AIRBORNE_PENALTY

        # Extract only active agents: [N, n_active]
        rewards = np.stack([rewards_full[:, idx] for idx in active], axis=1)

        # ── 9. Update step count and done ──────────────────────
        self.step_count += 1
        max_score = np.maximum(self.score[:, 0], self.score[:, 1])
        self.done = (self.step_count >= self.max_steps) | (max_score >= self.max_goals)

        # ── 9b. Draw penalty — punish all agents if episode times out as a draw
        timed_out = (self.step_count >= self.max_steps) & (self.score[:, 0] == self.score[:, 1])
        if timed_out.any():
            rewards -= timed_out[:, np.newaxis].astype(np.float32) * self.draw_penalty

        self._update_distances()

        # ── 10. Build info ─────────────────────────────────────
        done_snapshot = self.done.copy()

        infos = {
            "goal_team0": goal_team0,
            "goal_team1": goal_team1,
            "score": self.score.copy(),
            "episode_done": done_snapshot,
        }

        if self.done.any():
            self.reset(self.done)

        return self.get_obs(), rewards, done_snapshot, infos

    def _resolve_agent_agent_collisions(self):
        """Elastic collision between active agent pairs."""
        active = self.active_indices
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                i, j = active[ii], active[jj]
                diff = self.agent_pos[:, i] - self.agent_pos[:, j]
                dist = np.linalg.norm(diff, axis=-1, keepdims=True)
                overlap = 2 * AGENT_RADIUS - dist
                colliding = (overlap > 0).squeeze(-1)

                if not colliding.any():
                    continue

                normal = diff / (dist + 1e-8)
                sep = normal * overlap * 0.5
                self.agent_pos[colliding, i] += sep[colliding]
                self.agent_pos[colliding, j] -= sep[colliding]

                rel_vel = self.agent_vel[:, i] - self.agent_vel[:, j]
                vel_along_normal = (rel_vel * normal).sum(axis=-1, keepdims=True)
                impulse = normal * vel_along_normal * 0.5

                self.agent_vel[colliding, i] -= impulse[colliding]
                self.agent_vel[colliding, j] += impulse[colliding]

    def _resolve_agent_ball_collisions(self):
        """Separate overlapping agents and ball — no velocity transfer.

        Only kicks (action-driven) impart force to the ball. Body contact
        just prevents clipping through each other.
        """
        min_dist = AGENT_RADIUS + BALL_RADIUS
        for i in self.active_indices:
            diff = self.ball_pos - self.agent_pos[:, i]
            dist = np.linalg.norm(diff, axis=-1, keepdims=True)
            overlap = min_dist - dist
            colliding = (overlap > 0).squeeze(-1)

            if not colliding.any():
                continue

            # Position separation only — push apart proportional to mass
            normal = diff / (dist + 1e-8)
            total_mass = AGENT_MASS + BALL_MASS
            self.ball_pos[colliding] += (normal * overlap * AGENT_MASS / total_mass)[colliding]
            self.agent_pos[colliding, i] -= (normal * overlap * BALL_MASS / total_mass)[colliding]

    def _clamp_to_field_agents(self) -> np.ndarray:
        """Clamp active agents to field boundaries. Returns wall_clip mask [N, 4]."""
        r = AGENT_RADIUS
        pos = self.agent_pos

        hit_left = pos[:, :, 0] < r
        hit_right = pos[:, :, 0] > FIELD_W - r
        hit_bottom = pos[:, :, 1] < r
        hit_top = pos[:, :, 1] > FIELD_H - r
        wall_clip = hit_left | hit_right | hit_bottom | hit_top

        # Only clamp active agents
        for idx in self.active_indices:
            self.agent_pos[:, idx, 0] = np.clip(pos[:, idx, 0], r, FIELD_W - r)
            self.agent_pos[:, idx, 1] = np.clip(pos[:, idx, 1], r, FIELD_H - r)
            self.agent_vel[:, idx, 0] = np.where(
                hit_left[:, idx] | hit_right[:, idx],
                -self.agent_vel[:, idx, 0] * 0.3, self.agent_vel[:, idx, 0])
            self.agent_vel[:, idx, 1] = np.where(
                hit_bottom[:, idx] | hit_top[:, idx],
                -self.agent_vel[:, idx, 1] * 0.3, self.agent_vel[:, idx, 1])

        return wall_clip

    def _clamp_to_field_ball(self) -> np.ndarray:
        """Handle ball going out of bounds — throw-in for opposing team.

        Returns out_of_bounds [N] mask for reward computation.
        """
        r = BALL_RADIUS
        bx, by = self.ball_pos[:, 0], self.ball_pos[:, 1]

        # Side boundaries (top/bottom) — throw-in
        hit_bottom = by < r
        hit_top = by > FIELD_H - r
        side_out = hit_bottom | hit_top

        # End boundaries (left/right) outside goal — goal kick
        in_goal_y = (by >= GOAL_Y_MIN) & (by <= GOAL_Y_MAX)
        hit_left = (bx < r) & ~in_goal_y
        hit_right = (bx > FIELD_W - r) & ~in_goal_y
        end_out = hit_left | hit_right

        out_of_bounds = side_out | end_out

        if out_of_bounds.any():
            # Ball stops at boundary line; thrower placed outside in throw-in
            self.ball_pos[hit_bottom, 1] = r
            self.ball_pos[hit_top, 1] = FIELD_H - r
            self.ball_pos[hit_left, 0] = r
            self.ball_pos[hit_right, 0] = FIELD_W - r
            self.ball_vel[out_of_bounds] = 0.0
            self.ball_z[out_of_bounds] = 0.0
            self.ball_vz[out_of_bounds] = 0.0

        return out_of_bounds

    def _check_goals(self) -> tuple[np.ndarray, np.ndarray]:
        """Check if ball crossed goal lines (must be below goal height)."""
        bx, by = self.ball_pos[:, 0], self.ball_pos[:, 1]
        in_goal_y = (by >= GOAL_Y_MIN) & (by <= GOAL_Y_MAX)
        low_enough = self.ball_z < GOAL_HEIGHT
        team0_scored = (bx > FIELD_W) & in_goal_y & low_enough
        team1_scored = (bx < 0) & in_goal_y & low_enough
        return team0_scored, team1_scored

    def _compute_rewards(self, move: np.ndarray, goal_t0: np.ndarray, goal_t1: np.ndarray,
                         wall_clip: np.ndarray, kicked_ball: np.ndarray) -> np.ndarray:
        """Compute per-agent rewards. Returns [N, 4]."""
        rewards = np.zeros((self.n_envs, 4), dtype=np.float32)

        # Goal rewards — team 0 (agents 0,1), team 1 (agents 2,3)
        rewards[:, 0] += goal_t0 * self.goal_reward
        rewards[:, 1] += goal_t0 * self.goal_reward
        rewards[:, 0] -= goal_t1 * self.goal_reward
        rewards[:, 1] -= goal_t1 * self.goal_reward
        rewards[:, 2] += goal_t1 * self.goal_reward
        rewards[:, 3] += goal_t1 * self.goal_reward
        rewards[:, 2] -= goal_t0 * self.goal_reward
        rewards[:, 3] -= goal_t0 * self.goal_reward

        # Kick reward
        rewards += kicked_ball.astype(np.float32) * self.kick_reward

        # Approach ball reward
        ball_exp = self.ball_pos[:, np.newaxis, :]
        curr_ball_dist = np.linalg.norm(self.agent_pos - ball_exp, axis=-1)
        approach = self._prev_ball_dist - curr_ball_dist
        rewards += approach * self.approach_reward

        # Ball toward opponent goal
        curr_ball_goal_dist = np.linalg.norm(
            self.ball_pos - self.right_goal[np.newaxis, :], axis=-1
        )
        ball_progress = self._prev_ball_goal_dist - curr_ball_goal_dist
        rewards[:, 0] += ball_progress * self.ball_goal_reward
        rewards[:, 1] += ball_progress * self.ball_goal_reward
        rewards[:, 2] -= ball_progress * self.ball_goal_reward
        rewards[:, 3] -= ball_progress * self.ball_goal_reward

        # Energy penalty
        energy = np.linalg.norm(move, axis=-1)
        rewards -= energy * self.energy_penalty

        # Wall clip penalty
        rewards -= wall_clip.astype(np.float32) * self.wall_penalty

        # Ball-in-corner penalty — all agents penalized when ball is in a dead corner
        ball_to_corners = self.ball_pos[:, np.newaxis, :] - _CORNERS[np.newaxis, :, :]  # [N, 4, 2]
        corner_dists = np.linalg.norm(ball_to_corners, axis=-1)  # [N, 4]
        in_corner = (corner_dists.min(axis=1) < CORNER_RADIUS)   # [N]
        rewards -= in_corner[:, np.newaxis].astype(np.float32) * self.corner_ball_penalty

        # Ball-near-wall penalty — keeps play central, prevents wall-pinning
        bx, by = self.ball_pos[:, 0], self.ball_pos[:, 1]
        near_wall = (
            (bx < BALL_WALL_MARGIN) | (bx > FIELD_W - BALL_WALL_MARGIN) |
            (by < BALL_WALL_MARGIN) | (by > FIELD_H - BALL_WALL_MARGIN)
        )  # [N]
        rewards -= near_wall[:, np.newaxis].astype(np.float32) * self.ball_wall_penalty

        return rewards

    def _compute_skill_rewards(self, dist_to_ball: np.ndarray,
                               is_dribbling: np.ndarray,
                               is_juking: np.ndarray) -> np.ndarray:
        """Compute skill-based rewards: dribble, possession, juke. Returns [N, 4]."""
        rewards = np.zeros((self.n_envs, 4), dtype=np.float32)

        # Dribble reward: moving with ball under close control
        rewards += is_dribbling.astype(np.float32) * self.dribble_reward

        # Possession reward: having ball within 2m (weaker than dribble)
        has_possession = dist_to_ball < 2.0
        for idx in set(range(4)) - set(self.active_indices):
            has_possession[:, idx] = False
        rewards += has_possession.astype(np.float32) * self.possession_reward

        # Juke reward: sharp direction change while near ball and near opponent
        rewards += is_juking.astype(np.float32) * self.juke_reward

        return rewards

    def get_obs(self) -> np.ndarray:
        """Build observation tensor [N, n_active, OBS_SIZE]."""
        active = self.active_indices
        ppt = self.players_per_team
        obs = np.zeros((self.n_envs, self.n_active, OBS_SIZE), dtype=np.float32)

        for out_i, agent_idx in enumerate(active):
            team = 0 if agent_idx < 2 else 1

            # Determine indices for teammate, opp1, opp2
            if ppt == 2:
                ti = 1 if agent_idx == 0 else (0 if agent_idx == 1 else (3 if agent_idx == 2 else 2))
                oi1 = 2 if team == 0 else 0
                oi2 = 3 if team == 0 else 1
                has_teammate = True
                has_opp2 = True
            else:
                ti = None
                oi1 = 2 if team == 0 else 0
                oi2 = None
                has_teammate = False
                has_opp2 = False

            own_goal = self.left_goal if team == 0 else self.right_goal
            opp_goal = self.right_goal if team == 0 else self.left_goal

            pos_i = self.agent_pos[:, agent_idx].copy()
            vel_i = self.agent_vel[:, agent_idx].copy()

            if team == 1:
                pos_i[:, 0] = FIELD_W - pos_i[:, 0]
                vel_i[:, 0] = -vel_i[:, 0]

            def rel_pos(target_pos):
                p = target_pos.copy()
                if team == 1:
                    p[:, 0] = FIELD_W - p[:, 0]
                return p - pos_i

            def rel_vel(target_vel):
                v = target_vel.copy()
                if team == 1:
                    v[:, 0] = -v[:, 0]
                return v - vel_i

            norm_pos = pos_i.copy()
            norm_pos[:, 0] = (norm_pos[:, 0] / FIELD_W) * 2.0 - 1.0
            norm_pos[:, 1] = (norm_pos[:, 1] / FIELD_H) * 2.0 - 1.0
            norm_vel = vel_i / MAX_AGENT_SPEED

            o = 0
            obs[:, out_i, o:o+2] = norm_pos;   o += 2  # self pos
            obs[:, out_i, o:o+2] = norm_vel;    o += 2  # self vel

            # Teammate (zero if 1v1)
            if has_teammate:
                obs[:, out_i, o:o+2] = rel_pos(self.agent_pos[:, ti]) / 15.0
                o += 2
                obs[:, out_i, o:o+2] = rel_vel(self.agent_vel[:, ti]) / 5.0
                o += 2
            else:
                o += 4  # leave as zeros

            # Opponent 1 (always present)
            obs[:, out_i, o:o+2] = rel_pos(self.agent_pos[:, oi1]) / 15.0; o += 2
            obs[:, out_i, o:o+2] = rel_vel(self.agent_vel[:, oi1]) / 5.0;  o += 2

            # Opponent 2 (zero if 1v1)
            if has_opp2:
                obs[:, out_i, o:o+2] = rel_pos(self.agent_pos[:, oi2]) / 15.0
                o += 2
                obs[:, out_i, o:o+2] = rel_vel(self.agent_vel[:, oi2]) / 5.0
                o += 2
            else:
                o += 4  # leave as zeros

            # Ball
            obs[:, out_i, o:o+2] = rel_pos(self.ball_pos) / 15.0;  o += 2
            obs[:, out_i, o:o+2] = rel_vel(self.ball_vel) / 10.0;  o += 2

            # Goal positions
            own_g = np.broadcast_to(own_goal, (self.n_envs, 2)).copy()
            opp_g = np.broadcast_to(opp_goal, (self.n_envs, 2)).copy()
            obs[:, out_i, o:o+2] = rel_pos(own_g) / 15.0;  o += 2
            obs[:, out_i, o:o+2] = rel_pos(opp_g) / 15.0;  o += 2

            # Scalar features
            score_diff = (self.score[:, team] - self.score[:, 1 - team]).astype(np.float32) / self.max_goals
            step_frac = self.step_count.astype(np.float32) / self.max_steps
            team_flag = np.full(self.n_envs, 1.0 if team == 0 else -1.0, dtype=np.float32)

            # Closest to ball (among active agents only)
            active_dists = np.stack([
                np.linalg.norm(self.agent_pos[:, ai] - self.ball_pos, axis=-1)
                for ai in self.active_indices
            ], axis=1)  # [N, n_active]
            my_dist = np.linalg.norm(self.agent_pos[:, agent_idx] - self.ball_pos, axis=-1)
            closest = (my_dist == active_dists.min(axis=1)).astype(np.float32)

            obs[:, out_i, o] = score_diff;   o += 1
            obs[:, out_i, o] = step_frac;    o += 1
            obs[:, out_i, o] = team_flag;    o += 1
            obs[:, out_i, o] = closest;      o += 1

            # Has ball possession (within dribble range)
            has_ball = (my_dist < DRIBBLE_RANGE).astype(np.float32)
            obs[:, out_i, o] = has_ball;     o += 1

            # Ball speed (normalized)
            ball_spd = np.linalg.norm(self.ball_vel, axis=-1) / MAX_BALL_SPEED
            obs[:, out_i, o] = ball_spd;     o += 1

            # Ball height and vertical velocity
            obs[:, out_i, o] = self.ball_z / 3.0;   o += 1
            obs[:, out_i, o] = self.ball_vz / MAX_BALL_SPEED;  o += 1

            # Own height and vertical velocity (for jump awareness)
            obs[:, out_i, o] = self.agent_z[:, agent_idx] / 2.0;  o += 1
            obs[:, out_i, o] = self.agent_vz[:, agent_idx] / JUMP_IMPULSE;  o += 1

        return obs

    def get_display_state(self, env_idx: int = 0) -> dict:
        """Get state of a single env for visualization (only active agents)."""
        _contact_names = {0: None, 1: "foot", 2: "knee", 3: "header"}
        agents = []
        for i in self.active_indices:
            agents.append({
                "id": i,
                "team": 0 if i < 2 else 1,
                "x": float(self.agent_pos[env_idx, i, 0]),
                "y": float(self.agent_pos[env_idx, i, 1]),
                "vx": float(self.agent_vel[env_idx, i, 0]),
                "vy": float(self.agent_vel[env_idx, i, 1]),
                "facing": float(self.agent_facing[env_idx, i]),
                "kicking": float(self.agent_kick_state[env_idx, i]),
                "kick_leg": int(self._kick_leg[env_idx, i]),
                "dribbling": bool(self.agent_dribbling[env_idx, i]),
                "is_juking": bool(self.agent_juking[env_idx, i]),
                "kick_power": float(self.agent_kick_power[env_idx, i]),
                "contact_type": _contact_names[int(self.agent_contact_type[env_idx, i])],
                "z": float(self.agent_z[env_idx, i]),
            })
        return {
            "agents": agents,
            "ball": {
                "x": float(self.ball_pos[env_idx, 0]),
                "y": float(self.ball_pos[env_idx, 1]),
                "vx": float(self.ball_vel[env_idx, 0]),
                "vy": float(self.ball_vel[env_idx, 1]),
                "z": float(self.ball_z[env_idx]),
                "vz": float(self.ball_vz[env_idx]),
            },
            "score": [int(self.score[env_idx, 0]), int(self.score[env_idx, 1])],
            "step": int(self.step_count[env_idx]),
            "max_steps": int(self.max_steps),
            "max_goals": int(self.max_goals),
            "players_per_team": self.players_per_team,
        }
