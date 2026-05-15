"""Pure PyTorch PPO (Proximal Policy Optimization) for vectorized environments.

Supports multiple scenarios (soccer, drone) via env_class parameter.
"""

import io
import json as _json
import logging
import re
import threading
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

logger = logging.getLogger("deeprl.ppo")

MAX_SAVED_CHECKPOINTS = 30

# ── Network ────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """Shared-backbone actor-critic with continuous action output."""

    def __init__(self, obs_dim: int = 34, act_dim: int = 7, hidden: int = 512, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ELU(),
            ])
            in_dim = hidden
        self.shared = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden, 1)

        # Initialize actor output small for exploration
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        mean = torch.tanh(self.actor_mean(h))
        std = torch.clamp(self.log_std, -4.0, 1.0).exp()
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std, value = self(obs)
        if deterministic:
            action = mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


# ── Rollout Buffer ─────────────────────────────────────────────

class RolloutBuffer:
    """Stores rollout data for PPO update."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, actions, log_probs, rewards, values, dones):
        self.obs.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.values.append(values)
        self.dones.append(dones)

    def compute_returns(self, last_values: np.ndarray, gamma: float, gae_lambda: float):
        """Compute GAE advantages and returns."""
        n_steps = len(self.rewards)
        n_agents = self.rewards[0].shape[0]

        advantages = np.zeros(n_agents, dtype=np.float32)
        returns_list = []

        # Bootstrap from last values
        next_values = last_values.astype(np.float32)

        for t in reversed(range(n_steps)):
            rewards = self.rewards[t].astype(np.float32)
            values = self.values[t].astype(np.float32)
            dones = self.dones[t].astype(np.float32)

            # GAE delta
            delta = rewards + gamma * next_values * (1.0 - dones) - values
            advantages = delta + gamma * gae_lambda * (1.0 - dones) * advantages
            returns_list.insert(0, (advantages + values).astype(np.float32))
            next_values = values

        returns_arr = np.array(returns_list, dtype=np.float32)
        values_arr = np.array(self.values, dtype=np.float32)
        return returns_arr, (returns_arr - values_arr).astype(np.float32)

    def get_batches(self, returns: np.ndarray, advantages: np.ndarray, batch_size: int):
        """Yield shuffled minibatches."""
        # Flatten steps and agents — ensure float32
        obs = np.concatenate(self.obs, axis=0).astype(np.float32)
        actions = np.concatenate(self.actions, axis=0).astype(np.float32)
        log_probs = np.concatenate(self.log_probs, axis=0).astype(np.float32)
        ret = returns.reshape(-1).astype(np.float32)
        adv = advantages.reshape(-1).astype(np.float32)

        # Normalize advantages
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)

        n = obs.shape[0]
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield (
                obs[idx],
                actions[idx],
                log_probs[idx],
                ret[idx],
                adv[idx],
            )

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


# ── PPO Trainer ────────────────────────────────────────────────

class PPOTrainer(threading.Thread):
    """Runs PPO training in a background thread.

    Supports multiple scenarios via env/obs_size/action_size parameters.
    When scenario='soccer', tracks teams/goals. When scenario='drone', tracks waypoints/collisions.
    """

    def __init__(self, n_envs: int = 64, redis_client=None, players_per_team: int = 1,
                 scenario: str = "soccer", env=None, obs_size: int = None, action_size: int = None):
        super().__init__(daemon=True)
        self.n_envs = n_envs
        self.scenario = scenario

        # Redis key prefixes (namespaced per scenario)
        self._checkpoint_key_prefix = f"deeprl:{scenario}:checkpoint:saved:"
        self._checkpoint_index_key = f"deeprl:{scenario}:checkpoint:index"
        self._checkpoint_latest_key = f"deeprl:{scenario}:checkpoint:latest"

        # Create environment if not provided
        if env is not None:
            self.env = env
        elif scenario == "drone":
            from drone_environment import VectorizedDroneEnv
            self.env = VectorizedDroneEnv(n_envs=n_envs)
        else:
            from environment import VectorizedSoccerEnv
            self.env = VectorizedSoccerEnv(n_envs=n_envs, players_per_team=players_per_team)

        # Resolve obs/action sizes
        if obs_size is not None:
            self.obs_size = obs_size
        elif scenario == "drone":
            from drone_environment import OBS_SIZE as DRONE_OBS
            self.obs_size = DRONE_OBS
        else:
            from environment import OBS_SIZE as SOCCER_OBS
            self.obs_size = SOCCER_OBS

        if action_size is not None:
            self.action_size = action_size
        elif scenario == "drone":
            from drone_environment import ACTION_SIZE as DRONE_ACT
            self.action_size = DRONE_ACT
        else:
            from environment import ACTION_SIZE as SOCCER_ACT
            self.action_size = SOCCER_ACT

        # Hyperparameters (can be updated live)
        if scenario == "drone":
            self.n_steps = getattr(self.env, 'max_steps', 500)
            self.batch_size = 2048
            self.gamma = 0.995
            self.entropy_coeff = 0.005
        else:
            self.n_steps = 500  # synced to env.max_steps
            self.batch_size = 512
            self.gamma = 0.99
            self.entropy_coeff = 0.01
        self.n_epochs = 4
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.vf_coeff = 0.5
        self.max_grad_norm = 0.5
        self.lr = 1.5e-4

        # Training state
        self.generation = 0
        self.total_steps = 0
        self.episode_count = 0
        self._speed = 1  # 1x, 2x, 5x, 10x
        self._running = True
        self._training_active = True
        self._watch_mode = False
        self._loaded_checkpoint_name = None

        # Thread-safe state
        self._lock = threading.Lock()
        self._loop_idle = threading.Event()  # set when training loop is at top (safe to modify model)
        self._snapshot: dict | None = None
        self._snapshot_ready = threading.Event()
        self._stats: dict = {}
        self._episode_events: deque = deque(maxlen=100)

        # Pending team size change (applied in training loop, soccer only)
        self._pending_team_size = None

        # Manual flight mode (drone only)
        self._manual_mode = False
        self._manual_controls = None      # (throttle, pitch, roll, yaw) or None
        self._manual_recording = False
        self._demo_buffer = []            # list of (obs_1d, action_1d) numpy pairs
        self._pretrain_progress = None    # dict or None (broadcast by server)
        self._pre_manual_training_active = False
        self._pre_manual_watch_mode = False
        self._pretraining = False

        # Training session recorder (drone only)
        from recorder import TrainingRecorder
        self._recorder = TrainingRecorder()
        self._swarm_mode = False  # Send all env positions for swarm visualization

        # Episode tracking (all envs — for training metrics)
        self._episode_rewards_cyan = deque(maxlen=200)
        self._episode_rewards_magenta = deque(maxlen=200)
        self._episode_goals = deque(maxlen=200)
        self._episode_winners = deque(maxlen=200)

        # Drone-specific tracking
        self._episode_rewards = deque(maxlen=200)
        self._episode_gates = deque(maxlen=200)
        self._episode_completions = deque(maxlen=200)

        # Per-env per-team reward accumulators
        self._env_reward_acc_cyan = np.zeros(n_envs, dtype=np.float32)
        self._env_reward_acc_magenta = np.zeros(n_envs, dtype=np.float32)
        self._env_reward_acc = np.zeros(n_envs, dtype=np.float32)

        # Display env (env 0) episode counter — what the user sees
        self._display_episode_count = 0

        # Cumulative goal totals (across all episodes, soccer only)
        self._total_goals_cyan = 0
        self._total_goals_magenta = 0

        # Redis for checkpoints
        self._redis = redis_client

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PPO [{scenario}] using device: {self.device}")

        # Model
        self.model = ActorCritic(obs_dim=self.obs_size, act_dim=self.action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Try to load checkpoint from Redis
        self._load_checkpoint()

    def run(self):
        """Main training loop (runs in thread)."""
        if self.scenario == "soccer":
            logger.info(f"PPO [{self.scenario}] started: {self.n_envs} envs, {self.env.players_per_team}v{self.env.players_per_team} on {self.device}")
        else:
            logger.info(f"PPO [{self.scenario}] started: {self.n_envs} envs on {self.device}")
        obs = self.env.get_obs()  # [N, n_active, OBS_SIZE]

        while self._running:
            # Signal that the training loop is at its top — safe to modify model
            self._loop_idle.set()

            # Check for pending team size change
            if self._pending_team_size is not None:
                if self._apply_team_size_change():
                    obs = self.env.get_obs()

            if self._watch_mode:
                obs = self._run_watch_step(obs)
                time.sleep(0.02 / self._speed)
                continue

            if self._manual_mode:
                obs = self._run_manual_step(obs)
                time.sleep(0.02)  # real-time at 50Hz
                continue

            if not self._training_active:
                # Still publish display state when paused
                self._publish_snapshot(obs, 0)
                time.sleep(0.1)
                continue

            # ── Collect rollout ─────────────────────────────
            self._loop_idle.clear()  # model in use — not safe to modify
            buffer = RolloutBuffer()

            for step in range(self.n_steps):
                if not self._running:
                    return
                if self._manual_mode or self._watch_mode or not self._training_active:
                    break  # mode changed mid-rollout, abort and re-check

                n_active = self.env.n_active

                # Flatten: [N, n_active, OBS_SIZE] → [N*n_active, OBS_SIZE]
                obs_flat = obs.reshape(-1, self.obs_size)
                obs_t = torch.from_numpy(obs_flat).float().to(self.device)

                with torch.no_grad():
                    actions_t, log_probs_t, values_t = self.model.get_action(obs_t)

                actions_np = actions_t.cpu().numpy()    # [N*n_active, ACTION_SIZE]
                log_probs_np = log_probs_t.cpu().numpy()
                values_np = values_t.cpu().numpy()

                # Reshape actions back to [N, n_active, ACTION_SIZE]
                actions_env = actions_np.reshape(self.n_envs, n_active, self.action_size)

                # Step environment
                next_obs, rewards, dones, infos = self.env.step(actions_env)

                # Accumulate per-env rewards for episode return tracking
                if self.scenario == "soccer":
                    ppt = self.env.players_per_team
                    self._env_reward_acc_cyan += rewards[:, :ppt].mean(axis=1)
                    self._env_reward_acc_magenta += rewards[:, ppt:].mean(axis=1)
                    self._track_goals(infos)
                else:
                    self._env_reward_acc += rewards.mean(axis=1)

                # Track episodes (harvests accumulators on done)
                self._track_episodes(infos)

                # Store in buffer (flattened)
                rewards_flat = rewards.reshape(-1)      # [N*n_active]
                dones_flat = np.repeat(dones, n_active) # [N*n_active]

                buffer.add(obs_flat, actions_np, log_probs_np, rewards_flat, values_np, dones_flat)

                obs = next_obs
                self.total_steps += self.n_envs * n_active

                # Publish display state periodically (every ~50ms at 1x speed)
                if step % max(1, 4 // self._speed) == 0:
                    self._publish_snapshot(obs, step)

                # Speed control: sleep between steps for visualization
                if self._speed < 10:
                    time.sleep(0.02 / self._speed)

                # End generation early when display env's episode ends
                if dones[0]:
                    break

            # ── PPO Update ──────────────────────────────────
            # Skip update if mode changed mid-rollout
            if self._manual_mode or self._watch_mode or not self._training_active:
                continue

            # Mean episode returns (from recently completed episodes)
            if self.scenario == "soccer":
                avg_return_cyan = float(np.mean(self._episode_rewards_cyan)) if self._episode_rewards_cyan else 0.0
                avg_return_magenta = float(np.mean(self._episode_rewards_magenta)) if self._episode_rewards_magenta else 0.0

            # Check curriculum advance (drone only)
            if self.scenario == "drone" and hasattr(self.env, 'check_curriculum_advance'):
                old_level = getattr(self.env, 'curriculum_level', 1)
                old_endless = getattr(self.env, '_endless_mode', False)
                self.env.check_curriculum_advance()
                new_level = self.env.curriculum_level
                new_endless = getattr(self.env, '_endless_mode', False)
                if new_level != old_level or (new_endless and not old_endless):
                    self.n_steps = self.env.max_steps
                    if new_endless:
                        new_name = f"Endless #{self.env._endless_scenario}"
                    else:
                        from drone_environment import CURRICULUM_PROFILES
                        new_name = CURRICULUM_PROFILES.get(new_level, {}).get('name', '')
                    logger.info(
                        f"[drone] ★ CURRICULUM ADVANCE: level {old_level} → {new_level} "
                        f"({new_name}), n_steps={self.n_steps}"
                    )
                    # Record layout change for replay
                    self._recorder.record_layout(self.env.get_course_layout())
                    # Auto-save checkpoint at each curriculum milestone
                    if not new_endless:
                        self.save_named_checkpoint(f"level{new_level}")
                    else:
                        self.save_named_checkpoint("endless_start")

            # Bootstrap values for last step
            obs_flat = obs.reshape(-1, self.obs_size)
            obs_t = torch.from_numpy(obs_flat).float().to(self.device)
            with torch.no_grad():
                _, _, last_values = self.model(obs_t)
            last_values_np = last_values.cpu().numpy()

            returns, advantages = buffer.compute_returns(last_values_np, self.gamma, self.gae_lambda)

            # PPO epochs
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            n_batches = 0

            for epoch in range(self.n_epochs):
                for batch in buffer.get_batches(returns, advantages, self.batch_size):
                    b_obs, b_actions, b_old_log_probs, b_returns, b_advantages = batch

                    b_obs_t = torch.from_numpy(b_obs).float().to(self.device)
                    b_actions_t = torch.from_numpy(b_actions).float().to(self.device)
                    b_old_lp_t = torch.from_numpy(b_old_log_probs).float().to(self.device)
                    b_returns_t = torch.from_numpy(b_returns).float().to(self.device)
                    b_adv_t = torch.from_numpy(b_advantages).float().to(self.device)

                    new_log_probs, entropy, new_values = self.model.evaluate(b_obs_t, b_actions_t)

                    # Policy loss (clipped surrogate)
                    ratio = torch.exp(new_log_probs - b_old_lp_t)
                    surr1 = ratio * b_adv_t
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv_t
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = nn.functional.mse_loss(new_values, b_returns_t)

                    # Entropy bonus
                    entropy_loss = -entropy.mean()

                    # Total loss
                    loss = policy_loss + self.vf_coeff * value_loss + self.entropy_coeff * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += -entropy_loss.item()
                    n_batches += 1

            buffer.clear()

            self.generation += 1

            # Publish training stats
            avg_pl = total_policy_loss / max(n_batches, 1)
            avg_vl = total_value_loss / max(n_batches, 1)
            avg_ent = total_entropy / max(n_batches, 1)

            with self._lock:
                self._stats = {
                    "type": "training_stats",
                    "scenario": self.scenario,
                    "generation": self.generation,
                    "total_steps": self.total_steps,
                    "episodes": self._display_episode_count,
                    "policy_loss": round(avg_pl, 6),
                    "value_loss": round(avg_vl, 6),
                    "entropy": round(avg_ent, 4),
                }
                if self.scenario == "soccer":
                    wr_cyan, wr_magenta = self._compute_win_rate()
                    self._stats.update({
                        "avg_reward_cyan": round(avg_return_cyan, 4),
                        "avg_reward_magenta": round(avg_return_magenta, 4),
                        "win_rate_cyan": wr_cyan,
                        "win_rate_magenta": wr_magenta,
                        "goals_per_episode_avg": round(float(np.mean(self._episode_goals)) if self._episode_goals else 0, 2),
                    })
                else:
                    avg_reward = float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0
                    avg_gates = float(np.mean(self._episode_gates)) if self._episode_gates else 0.0
                    completion_rate = float(np.mean(self._episode_completions)) if self._episode_completions else 0.0
                    self._stats.update({
                        "avg_reward": round(avg_reward, 4),
                        "avg_gates_passed": round(avg_gates, 2),
                        "completion_rate": round(completion_rate, 3),
                        "curriculum_level": getattr(self.env, 'curriculum_level', 1),
                    })

            self._recorder.record_stats(self._stats)

            # Save checkpoint periodically
            if self.generation % 10 == 0:
                self._save_checkpoint()

            if self.scenario == "soccer":
                logger.info(
                    f"[soccer] Gen {self.generation}: pl={avg_pl:.4f} vl={avg_vl:.4f} "
                    f"ent={avg_ent:.3f} cyan={avg_return_cyan:.2f} mag={avg_return_magenta:.2f} "
                    f"steps={self.total_steps}"
                )
            else:
                avg_r = float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0
                comp_r = float(np.mean(self._episode_completions)) if self._episode_completions else 0.0
                logger.info(
                    f"[drone] Gen {self.generation}: pl={avg_pl:.4f} vl={avg_vl:.4f} "
                    f"ent={avg_ent:.3f} reward={avg_r:.2f} comp={comp_r:.1%} "
                    f"steps={self.total_steps}"
                )

    def _track_episodes(self, infos: dict):
        """Track episode completion events."""
        done_mask = infos["episode_done"]
        if not done_mask.any():
            return

        if self.scenario == "soccer":
            self._track_episodes_soccer(infos, done_mask)
        else:
            self._track_episodes_drone(infos, done_mask)

    def _track_episodes_soccer(self, infos, done_mask):
        """Soccer-specific episode tracking."""
        done_indices = np.where(done_mask)[0]
        for idx in done_indices:
            self._episode_rewards_cyan.append(float(self._env_reward_acc_cyan[idx]))
            self._episode_rewards_magenta.append(float(self._env_reward_acc_magenta[idx]))
        self._env_reward_acc_cyan[done_mask] = 0.0
        self._env_reward_acc_magenta[done_mask] = 0.0

        scores = infos["score"][done_mask]
        for i in range(scores.shape[0]):
            s0, s1 = scores[i]
            total_goals = s0 + s1
            self._episode_goals.append(total_goals)
            self.episode_count += 1

            if s0 > s1:
                winner = "cyan"
            elif s1 > s0:
                winner = "magenta"
            else:
                winner = "draw"
            self._episode_winners.append(winner)

        if done_mask[0]:
            s0, s1 = infos["score"][0]
            self._display_episode_count += 1
            if s0 > s1:
                winner = "cyan"
            elif s1 > s0:
                winner = "magenta"
            else:
                winner = "draw"
            with self._lock:
                self._episode_events.append({
                    "type": "episode_end",
                    "episode": self._display_episode_count,
                    "winner": winner,
                    "final_score": [int(s0), int(s1)],
                    "duration_steps": int(self.env.step_count[0]),
                })

    def _track_episodes_drone(self, infos, done_mask):
        """Drone-specific episode tracking."""
        done_indices = np.where(done_mask)[0]
        for idx in done_indices:
            self._episode_rewards.append(float(self._env_reward_acc[idx]))
            self._episode_gates.append(int(infos['gates_passed'][idx]))
            self._episode_completions.append(bool(infos.get('completed', np.zeros(self.n_envs, dtype=bool))[idx]))
        self._env_reward_acc[done_mask] = 0.0
        self.episode_count += done_mask.sum()

        if done_mask[0]:
            self._display_episode_count += 1
            event = {
                "type": "episode_end",
                "episode": self._display_episode_count,
                "gates_passed": int(infos['gates_passed'][0]),
                "total_gates": int(self.env.total_gates),
                "completed": bool(infos.get('completed', np.zeros(self.n_envs, dtype=bool))[0]),
                "duration_steps": int(self.env.step_count[0]),
            }
            self._recorder.record_event(event)
            with self._lock:
                self._episode_events.append(event)

    def _track_goals(self, infos: dict):
        """Emit goal_scored event for the display env (env 0) when a goal happens."""
        g0 = infos["goal_team0"][0]  # bool — did team 0 score in env 0?
        g1 = infos["goal_team1"][0]
        if g0 or g1:
            if g0:
                self._total_goals_cyan += 1
            if g1:
                self._total_goals_magenta += 1
            team = "cyan" if g0 else "magenta"
            with self._lock:
                self._episode_events.append({
                    "type": "goal_scored",
                    "team": team,
                    "total_score": [self._total_goals_cyan, self._total_goals_magenta],
                })

    def _run_watch_step(self, obs: np.ndarray) -> np.ndarray:
        """Run one env step with deterministic policy (no training)."""
        n_active = self.env.n_active
        obs_flat = obs.reshape(-1, self.obs_size)
        obs_t = torch.from_numpy(obs_flat).float().to(self.device)

        with torch.no_grad():
            actions_t, _, _ = self.model.get_action(obs_t, deterministic=True)

        actions_np = actions_t.cpu().numpy()
        actions_env = actions_np.reshape(self.n_envs, n_active, self.action_size)
        next_obs, rewards, dones, infos = self.env.step(actions_env)

        if self.scenario == "soccer":
            self._track_goals(infos)
        self._track_episodes(infos)
        self._publish_snapshot(next_obs, 0)
        return next_obs

    def set_watch_mode(self):
        """Enter watch mode: deterministic inference, no training."""
        self._training_active = False
        self._watch_mode = True

    def set_train_mode(self):
        """Enter training mode (resume or start)."""
        self._watch_mode = False
        self._manual_mode = False
        self._training_active = True

    # ── Manual Flight Mode (drone only) ──────────────────────────

    def set_manual_mode(self, enabled: bool):
        """Enter/exit manual flight mode for human demonstration."""
        if enabled:
            # Save prior state so we can restore on exit
            self._pre_manual_training_active = self._training_active
            self._pre_manual_watch_mode = self._watch_mode
            self._training_active = False
            self._watch_mode = False
            self._manual_mode = True
            self._manual_controls = None
            # Reset env 0 so human starts from clean state
            self.env.reset(env_mask=np.array([True] + [False] * (self.n_envs - 1)))
            self._calm_manual_env()
        else:
            self._manual_mode = False
            self._manual_recording = False
            # Restore prior state
            self._training_active = self._pre_manual_training_active
            self._watch_mode = self._pre_manual_watch_mode

    def set_manual_controls(self, throttle: float, pitch: float, roll: float, yaw: float):
        """Set raw human control inputs (not motor values)."""
        self._manual_controls = (throttle, pitch, roll, yaw)

    def _calm_manual_env(self):
        """Force env 0 to nominal, calm conditions for predictable manual flight.

        Eliminates wind, turbulence, gusts, and domain randomization so the
        human pilot experiences consistent, deterministic physics. Called before
        every manual step to counteract auto-reset re-randomization.
        """
        from drone_environment import MASS, IXX, IYY, IZZ, KF, KM, MOTOR_TAU
        e = self.env
        # Zero all wind effects for env 0
        e.episode_wind[0] = 0.0
        e.base_wind[0] = 0.0
        e.turb_state[0] = 0.0
        e.turb_intensity[0] = 0.0
        e.wind_force[0] = 0.0
        # Block gust generation: active gust with zero strength prevents
        # the random gust spawn branch from firing
        e.gust_remaining[0] = 999.0
        e.gust_strength[0] = 0.0
        # Nominal physics (no domain randomization)
        e.dr_mass[0] = MASS
        e.dr_ixx[0] = IXX
        e.dr_iyy[0] = IYY
        e.dr_izz[0] = IZZ
        e.dr_kf[0] = KF
        e.dr_km[0] = KM
        e.dr_motor_tau[0] = MOTOR_TAU
        # Disable projectiles and EW for env 0
        e.proj_active[0] = False
        e.proj_spawn_timer[0] = 99999.0
        e.gps_denied[0] = False
        e.jamming_intensity[0] = 0.0
        # Prevent episode time-limit termination (human flies as long as they want)
        e.step_count[0] = 0

    def _run_manual_step(self, obs: np.ndarray) -> np.ndarray:
        """Step env with human actions on env 0, model-hover on others.

        Angle-mode flight controller (like Betaflight angle mode):
        - Human pitch/roll inputs (±1) set target tilt angles
        - Human yaw input (±1) sets target yaw rate
        - Human throttle (±1) sets thrust offset
        - PD controller drives drone to target attitudes/rates
        - Releasing keys → targets go to zero → drone self-levels
        """
        from drone_environment import HOVER_OMEGA, MAX_OMEGA
        hover_base = (HOVER_OMEGA / MAX_OMEGA) * 2.0 - 1.0

        # Force clean conditions for env 0 every step (counteracts auto-reset
        # re-randomization if the drone crashed or went OOB last step)
        self._calm_manual_env()

        # ── Flight controller tuning ─────────────────────
        MAX_TILT = 0.26       # 15° max tilt angle (rad)
        MAX_YAW_RATE = 1.5    # max yaw rate (rad/s)
        THROTTLE_SCALE = 0.08 # motor offset for full throttle key
        KP = 0.15             # attitude P gain
        KD = 0.06             # attitude D gain (rate damping)
        KD_YAW = 0.04         # yaw rate gain

        # Human control inputs: normalized ±1 (or zeros = hover)
        if self._manual_controls is not None:
            h_thr, h_pitch, h_roll, h_yaw = self._manual_controls
        else:
            h_thr, h_pitch, h_roll, h_yaw = 0.0, 0.0, 0.0, 0.0

        # Map to physical targets
        target_pitch = h_pitch * MAX_TILT         # ±15° (neg = nose down = forward)
        target_roll  = h_roll * MAX_TILT           # ±15° (pos = tilt right)
        target_yaw_rate = -h_yaw * MAX_YAW_RATE    # sign flip: +input → CW → neg ang_vel
        throttle = h_thr * THROTTLE_SCALE

        # Read drone state directly from environment (not obs, which may be
        # stale after env reset and would cause phantom PD corrections → drift)
        ang_vel = self.env.ang_vel[0].copy()   # body-frame angular velocity (rad/s)
        quat = self.env.quat[0]                # [w, x, y, z]
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        euler_roll  = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        euler_pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

        # PD controller: drive to target angles / rates
        # Sign conventions (verified against physics + mixer):
        #   Mixer: +pitch → nose down, +roll → tilt left
        #   Euler: +pitch → nose up,   +roll → tilted right
        #   These are opposite, so error = (euler - target) naturally corrects.
        final_pitch = KP * (euler_pitch - target_pitch) + KD * ang_vel[1]
        final_roll  = KP * (euler_roll  - target_roll)  + KD * ang_vel[0]
        final_yaw   = KD_YAW * (ang_vel[2] - target_yaw_rate)

        # X-config motor mixing
        m0 = hover_base + throttle - final_pitch + final_roll - final_yaw  # FR CW
        m1 = hover_base + throttle - final_pitch - final_roll + final_yaw  # FL CCW
        m2 = hover_base + throttle + final_pitch - final_roll - final_yaw  # BL CW
        m3 = hover_base + throttle + final_pitch + final_roll + final_yaw  # BR CCW
        human_act = np.clip(np.array([m0, m1, m2, m3], dtype=np.float32), -1.0, 1.0)

        # Build action array: stabilized human on env 0, hover on rest
        n_active = self.env.n_active
        hover = self._hover_actions()
        actions_all = np.full((self.n_envs, n_active, self.action_size), 0.0, dtype=np.float32)
        for i in range(self.n_envs):
            actions_all[i, 0, :] = human_act if i == 0 else hover

        next_obs, rewards, dones, infos = self.env.step(actions_all)

        # Record stabilized motor output (not raw human input) so BC
        # learns the full flight-controller-assisted behavior
        if self._manual_recording:
            self._demo_buffer.append((
                obs[0, 0].copy().astype(np.float32),
                human_act.copy(),
            ))

        self._track_episodes(infos)
        self._publish_snapshot(next_obs, 0)
        return next_obs

    def _hover_actions(self) -> np.ndarray:
        """Return action array producing hover thrust."""
        from drone_environment import HOVER_OMEGA, MAX_OMEGA
        hover_act = (HOVER_OMEGA / MAX_OMEGA) * 2.0 - 1.0
        return np.full(self.action_size, hover_act, dtype=np.float32)

    def start_demo_recording(self):
        self._manual_recording = True
        self._demo_buffer.clear()

    def stop_demo_recording(self):
        self._manual_recording = False

    def get_demo_buffer_length(self) -> int:
        return len(self._demo_buffer)

    def save_demo(self, name: str) -> dict:
        """Save recorded demo buffer to Redis."""
        if not self._redis:
            return {"error": "Redis not available"}
        if not self._demo_buffer:
            return {"error": "No recorded data"}
        if not re.match(r'^[a-zA-Z0-9_-]{1,40}$', name):
            return {"error": "Name must be 1-40 alphanumeric/hyphen/underscore characters"}

        obs_arr = np.array([p[0] for p in self._demo_buffer], dtype=np.float32)
        act_arr = np.array([p[1] for p in self._demo_buffer], dtype=np.float32)

        # Serialize with numpy savez
        buf = io.BytesIO()
        np.savez_compressed(buf, obs=obs_arr, actions=act_arr)
        data = buf.getvalue()

        key = f"deeprl:drone:demo:data:{name}"
        self._redis.set(key, data)

        meta = {
            "name": name,
            "steps": len(self._demo_buffer),
            "duration": round(len(self._demo_buffer) * 0.02, 1),  # seconds at 50Hz
            "level": getattr(self.env, 'curriculum_level', 1),
            "timestamp": time.time(),
        }
        self._redis.hset("deeprl:drone:demo:index", name, _json.dumps(meta))

        logger.info(f"Demo saved: '{name}' ({meta['steps']} steps, {meta['duration']}s)")
        self._demo_buffer.clear()
        return meta

    def list_demos(self) -> list[dict]:
        """List all saved demos."""
        if not self._redis:
            return []
        try:
            raw = self._redis.hgetall("deeprl:drone:demo:index")
            demos = []
            for name_bytes, meta_json in raw.items():
                name_str = name_bytes.decode() if isinstance(name_bytes, bytes) else name_bytes
                meta = _json.loads(meta_json)
                meta["name"] = name_str
                demos.append(meta)
            demos.sort(key=lambda d: d.get("timestamp", 0), reverse=True)
            return demos
        except Exception as e:
            logger.warning(f"Failed to list demos: {e}")
            return []

    def delete_demo(self, name: str) -> dict:
        if not self._redis:
            return {"error": "Redis not available"}
        self._redis.delete(f"deeprl:drone:demo:data:{name}")
        self._redis.hdel("deeprl:drone:demo:index", name)
        logger.info(f"Demo deleted: '{name}'")
        return {"status": "deleted", "name": name}

    # ── Training Session Recording ──────────────────────────────

    def start_recording(self, name: str = ""):
        """Start recording the training session for later replay."""
        import time as _time
        if not name:
            name = _time.strftime("session-%Y%m%d-%H%M%S")
        self._recorder.start(name)
        # Record initial course layout so replay can build the scene
        if hasattr(self.env, 'get_course_layout'):
            self._recorder.record_layout(self.env.get_course_layout())

    def stop_recording(self) -> dict:
        """Stop recording, save to Redis, return metadata."""
        if not self._recorder.active:
            return {"error": "Not recording"}
        metadata, blob = self._recorder.stop()
        if self._redis:
            self._redis.set(f"deeprl:drone:recording:data:{metadata['name']}", blob)
            self._redis.hset("deeprl:drone:recording:index",
                             metadata['name'], _json.dumps(metadata))
        return metadata

    def list_recordings(self) -> list[dict]:
        """List all saved training recordings."""
        if not self._redis:
            return []
        index = self._redis.hgetall("deeprl:drone:recording:index")
        recordings = []
        for name, meta_json in index.items():
            name = name.decode() if isinstance(name, bytes) else name
            meta_json = meta_json.decode() if isinstance(meta_json, bytes) else meta_json
            recordings.append(_json.loads(meta_json))
        recordings.sort(key=lambda r: r.get("recorded_at", 0), reverse=True)
        return recordings

    def load_recording(self, name: str) -> bytes | None:
        """Load recording blob from Redis."""
        if not self._redis:
            return None
        return self._redis.get(f"deeprl:drone:recording:data:{name}")

    def delete_recording(self, name: str) -> dict:
        """Delete a saved recording."""
        if not self._redis:
            return {"error": "Redis not available"}
        self._redis.delete(f"deeprl:drone:recording:data:{name}")
        self._redis.hdel("deeprl:drone:recording:index", name)
        logger.info(f"Recording deleted: '{name}'")
        return {"status": "deleted", "name": name}

    @property
    def is_recording(self) -> bool:
        return self._recorder.active

    def pretrain_from_demos(self, demo_names: list[str], epochs: int = 50, lr: float = 1e-3):
        """Behavioral cloning: MSE loss to imitate human demonstrations.

        Only updates actor weights (shared backbone + actor_mean).
        Critic and log_std are untouched so PPO can fine-tune properly after.
        """
        if not self._redis:
            return {"error": "Redis not available"}
        if self._pretraining:
            return {"error": "Already pretraining"}

        # Load and concatenate all demos
        all_obs, all_act = [], []
        for name in demo_names:
            data = self._redis.get(f"deeprl:drone:demo:data:{name}")
            if not data:
                return {"error": f"Demo '{name}' not found"}
            buf = io.BytesIO(data)
            npz = np.load(buf)
            all_obs.append(npz['obs'])
            all_act.append(npz['actions'])

        obs_np = np.concatenate(all_obs, axis=0).astype(np.float32)
        act_np = np.concatenate(all_act, axis=0).astype(np.float32)
        n_samples = obs_np.shape[0]

        if n_samples < 10:
            return {"error": f"Too few samples ({n_samples}). Record longer demos."}

        logger.info(f"Behavioral cloning: {n_samples} samples from {len(demo_names)} demos, {epochs} epochs")

        # Pause training during BC
        was_training = self._training_active
        self._training_active = False
        self._watch_mode = False
        self._manual_mode = False
        self._pretraining = True
        if not self._loop_idle.wait(timeout=5.0):
            logger.warning("Timed out waiting for training loop idle before BC")
        self._loop_idle.clear()

        obs_t = torch.from_numpy(obs_np).to(self.device)
        act_t = torch.from_numpy(act_np).to(self.device)

        # Freeze critic: only train shared backbone + actor_mean
        for p in self.model.critic.parameters():
            p.requires_grad = False
        self.model.log_std.requires_grad = False

        bc_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )

        batch_size = min(512, n_samples)
        best_loss = float('inf')

        try:
            for epoch in range(epochs):
                if not self._running or not self._pretraining:
                    break

                indices = np.random.permutation(n_samples)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    idx = indices[start:end]

                    b_obs = obs_t[idx]
                    b_act = act_t[idx]

                    # Forward: get actor mean (tanh already applied in forward())
                    mean, _, _ = self.model(b_obs)
                    loss = nn.functional.mse_loss(mean, b_act)

                    bc_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    bc_optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                best_loss = min(best_loss, avg_loss)

                with self._lock:
                    self._pretrain_progress = {
                        "type": "pretrain_progress",
                        "epoch": epoch + 1,
                        "total_epochs": epochs,
                        "loss": round(avg_loss, 6),
                        "best_loss": round(best_loss, 6),
                    }

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"BC epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        finally:
            # Unfreeze critic + log_std for PPO
            for p in self.model.critic.parameters():
                p.requires_grad = True
            self.model.log_std.requires_grad = True
            self._pretraining = False

        logger.info(f"Behavioral cloning complete: final_loss={best_loss:.6f}")

        # ── Post-BC: make the effect visible and preserve it ────

        # 1. Reduce exploration noise so PPO starts near the demo behavior
        #    instead of drowning the learned mean in N(0, 1.0) noise.
        #    std=0.3 (log_std=-1.2) gives enough exploration to surpass
        #    human performance while keeping the BC policy recognizable.
        with torch.no_grad():
            self.model.log_std.fill_(-1.2)
        logger.info("Post-BC: log_std set to -1.2 (std=0.30)")

        # 2. Reset PPO optimizer so stale Adam momentum from any prior
        #    training doesn't push weights away from the BC solution.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        logger.info("Post-BC: PPO optimizer reset (fresh Adam state)")

        # 3. Resume training so PPO immediately fine-tunes from the BC policy.
        #    The BC effect is visible right away (drone starts competent, not random).
        self._training_active = True
        self._watch_mode = False
        self._manual_mode = False
        logger.info("Post-BC: resumed training — PPO fine-tuning from BC policy")

        # Signal completion
        with self._lock:
            self._pretrain_progress = {
                "type": "pretrain_progress",
                "epoch": epochs,
                "total_epochs": epochs,
                "loss": round(best_loss, 6),
                "best_loss": round(best_loss, 6),
                "done": True,
            }

        return {"status": "done", "epochs": epochs, "final_loss": round(best_loss, 6)}

    def get_pretrain_progress(self) -> dict | None:
        """Get and clear BC training progress for broadcast."""
        with self._lock:
            prog = self._pretrain_progress
            self._pretrain_progress = None
            return prog

    def _compute_win_rate(self) -> tuple[float, float]:
        if not self._episode_winners:
            return 0.5, 0.5
        total = len(self._episode_winners)
        cyan_wins = sum(1 for w in self._episode_winners if w == "cyan")
        magenta_wins = sum(1 for w in self._episode_winners if w == "magenta")
        return round(cyan_wins / total, 3), round(magenta_wins / total, 3)

    def _publish_snapshot(self, obs: np.ndarray, step: int):
        """Store latest display state for the broadcast loop."""
        state = self.env.get_display_state(0)
        state["episode"] = self._display_episode_count
        state["generation"] = self.generation
        state["total_score"] = [self._total_goals_cyan, self._total_goals_magenta]
        state["type"] = "state"

        # Swarm visualization: add all env positions
        if self._swarm_mode and self.scenario == "drone":
            state["swarm"] = self.env.get_swarm_state()

        self._recorder.record_state(state)

        with self._lock:
            self._snapshot = state
            self._snapshot_ready.set()

    def get_snapshot(self) -> dict | None:
        """Called by the async broadcast loop. Returns latest snapshot or None."""
        with self._lock:
            snap = self._snapshot
            self._snapshot = None
            self._snapshot_ready.clear()
            return snap

    def get_stats(self) -> dict:
        """Get latest training stats."""
        with self._lock:
            return self._stats.copy()

    def get_episode_events(self) -> list:
        """Drain pending episode events."""
        with self._lock:
            events = list(self._episode_events)
            self._episode_events.clear()
            return events

    def get_status(self) -> dict:
        status = {
            "type": "status",
            "scenario": self.scenario,
            "training": self._training_active,
            "watch_mode": self._watch_mode,
            "manual_mode": self._manual_mode,
            "recording": self._manual_recording,
            "pretraining": self._pretraining,
            "generation": self.generation,
            "total_steps": self.total_steps,
            "episode_count": self._display_episode_count,
            "speed": self._speed,
            "device": str(self.device),
            "n_envs": self.n_envs,
            "loaded_checkpoint": self._loaded_checkpoint_name,
        }
        if self.scenario == "soccer":
            status["players_per_team"] = self.env.players_per_team
        elif self.scenario == "drone":
            level = getattr(self.env, 'curriculum_level', 1)
            status["curriculum_level"] = level
            from drone_environment import CURRICULUM_PROFILES
            status["level_name"] = CURRICULUM_PROFILES.get(level, {}).get('name', '')
            status["obs_size"] = self.obs_size
            status["action_size"] = self.action_size
            status["session_recording"] = self._recorder.active
        return status

    def set_speed(self, speed: int):
        self._speed = max(1, min(10, speed))

    def set_swarm_mode(self, enabled: bool):
        self._swarm_mode = enabled

    def start_training(self):
        self._training_active = True

    def stop_training(self):
        self._training_active = False
        self._watch_mode = False

    def set_team_size(self, players_per_team: int):
        """Request switch between 1v1 and 2v2. Applied at next safe point in training loop."""
        if players_per_team not in (1, 2):
            return
        self._pending_team_size = players_per_team

    def _apply_team_size_change(self):
        """Apply pending team size change (called from training loop thread, soccer only)."""
        ppt = self._pending_team_size
        self._pending_team_size = None
        if ppt is None or self.scenario != "soccer" or ppt == self.env.players_per_team:
            return False
        self.env.set_players_per_team(ppt)
        self.model = ActorCritic(obs_dim=self.obs_size, act_dim=self.action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.generation = 0
        self.total_steps = 0
        self.episode_count = 0
        self._display_episode_count = 0
        self._env_reward_acc_cyan[:] = 0.0
        self._env_reward_acc_magenta[:] = 0.0
        self._episode_rewards_cyan.clear()
        self._episode_rewards_magenta.clear()
        self._episode_goals.clear()
        self._episode_winners.clear()
        self._total_goals_cyan = 0
        self._total_goals_magenta = 0
        self._loaded_checkpoint_name = None
        self._stats = {}
        logger.info(f"Team size changed to {ppt}v{ppt}")
        return True

    def reset_training(self):
        """Reset to random policy and start training."""
        self._watch_mode = False
        self._training_active = True
        self._loaded_checkpoint_name = None
        with self._lock:
            self.model = ActorCritic(obs_dim=self.obs_size, act_dim=self.action_size).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.generation = 0
            self.total_steps = 0
            self.episode_count = 0
            self._display_episode_count = 0
            self._env_reward_acc_cyan[:] = 0.0
            self._env_reward_acc_magenta[:] = 0.0
            self._env_reward_acc[:] = 0.0
            self._episode_rewards_cyan.clear()
            self._episode_rewards_magenta.clear()
            self._episode_goals.clear()
            self._episode_winners.clear()
            self._episode_rewards.clear()
            self._episode_gates.clear()
            self._episode_completions.clear()
            self._total_goals_cyan = 0
            self._total_goals_magenta = 0
            self._stats = {}
            self.env.reset()

    def update_config(self, config: dict):
        """Update hyperparameters live."""
        if "learning_rate" in config:
            self.lr = float(config["learning_rate"])
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lr
        if "entropy_coeff" in config:
            self.entropy_coeff = float(config["entropy_coeff"])

        if self.scenario == "soccer":
            # Soccer reward/penalty weights → environment
            env_keys = {
                "goal_reward": "goal_reward",
                "approach_reward": "approach_reward",
                "ball_goal_reward": "ball_goal_reward",
                "kick_reward": "kick_reward",
                "draw_penalty": "draw_penalty",
                "energy_penalty": "energy_penalty",
                "wall_penalty": "wall_penalty",
                "corner_ball_penalty": "corner_ball_penalty",
                "ball_wall_penalty": "ball_wall_penalty",
                "dribble_reward": "dribble_reward",
                "possession_reward": "possession_reward",
                "juke_reward": "juke_reward",
            }
            for cfg_key, env_attr in env_keys.items():
                if cfg_key in config:
                    setattr(self.env, env_attr, float(config[cfg_key]))
            if "max_goals" in config:
                self.env.max_goals = int(config["max_goals"])
        elif self.scenario == "drone":
            # Drone reward weights → environment
            drone_keys = {
                "waypoint_reward": "waypoint_reward",
                "progress_reward": "progress_reward",
                "course_complete_reward": "course_complete_reward",
                "survival_reward": "survival_reward",
                "collision_penalty": "collision_penalty",
                "crash_penalty": "crash_penalty",
                "oob_penalty": "oob_penalty",
                "projectile_hit_penalty": "projectile_hit_penalty",
                "dodge_bonus": "dodge_bonus",
                "stability_coeff": "stability_coeff",
                "energy_coeff": "energy_coeff",
                "smoothness_coeff": "smoothness_coeff",
                "altitude_coeff": "altitude_coeff",
                "orientation_coeff": "orientation_coeff",
                "speed_coeff": "speed_coeff",
                "gate_align_bonus": "gate_align_bonus",
                "drift_coeff": "drift_coeff",
            }
            for cfg_key, env_attr in drone_keys.items():
                if cfg_key in config:
                    setattr(self.env, env_attr, float(config[cfg_key]))
            if "curriculum_level" in config:
                self.env.set_curriculum_level(int(config["curriculum_level"]))
                # Sync rollout length to new level's max_steps
                self.n_steps = self.env.max_steps

        # Episode length (integer) — sync rollout length to match
        if "max_steps" in config:
            self.env.max_steps = int(config["max_steps"])
            self.n_steps = int(config["max_steps"])

    # ── Named Checkpoint Management ─────────────────────────────

    def save_named_checkpoint(self, name: str) -> dict:
        """Save a named checkpoint with metadata."""
        if not self._redis:
            return {"error": "Redis not available"}
        if not re.match(r'^[a-zA-Z0-9_-]{1,40}$', name):
            return {"error": "Name must be 1-40 alphanumeric/hyphen/underscore characters"}

        existing = self._redis.hlen(self._checkpoint_index_key)
        if existing >= MAX_SAVED_CHECKPOINTS:
            if not self._redis.hexists(self._checkpoint_index_key, name):
                return {"error": f"Maximum {MAX_SAVED_CHECKPOINTS} checkpoints. Delete one first."}

        try:
            buf = io.BytesIO()
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "generation": self.generation,
                "total_steps": self.total_steps,
                "episode_count": self.episode_count,
                "display_episode_count": self._display_episode_count,
                "players_per_team": getattr(self.env, 'players_per_team', 1),
                "action_size": self.action_size,
                "obs_size": self.obs_size,
                "scenario": self.scenario,
                "curriculum_level": getattr(self.env, 'curriculum_level', 1),
            }, buf)
            self._redis.set(f"{self._checkpoint_key_prefix}{name}", buf.getvalue())

            meta = {
                "name": name,
                "scenario": self.scenario,
                "generation": self.generation,
                "total_steps": self.total_steps,
                "players_per_team": getattr(self.env, 'players_per_team', 1),
                "episode_count": self._display_episode_count,
                "timestamp": time.time(),
            }
            if self.scenario == "soccer":
                meta["win_rate"] = self._compute_win_rate()[0]
            elif self.scenario == "drone":
                meta["completion_rate"] = float(np.mean(self._episode_completions)) if self._episode_completions else 0.0
                meta["curriculum_level"] = getattr(self.env, 'curriculum_level', 1)
            self._redis.hset(self._checkpoint_index_key, name, _json.dumps(meta))
            self._loaded_checkpoint_name = name
            logger.info(f"Named checkpoint saved: {name} (gen={self.generation})")
            return meta
        except Exception as e:
            logger.warning(f"Failed to save named checkpoint: {e}")
            return {"error": str(e)}

    def list_checkpoints(self) -> list[dict]:
        """List all saved checkpoints with metadata."""
        if not self._redis:
            return []
        try:
            raw = self._redis.hgetall(self._checkpoint_index_key)
            checkpoints = []
            for name_bytes, meta_json in raw.items():
                name_str = name_bytes.decode() if isinstance(name_bytes, bytes) else name_bytes
                meta = _json.loads(meta_json)
                meta["name"] = name_str
                meta["active"] = (name_str == self._loaded_checkpoint_name)
                checkpoints.append(meta)
            checkpoints.sort(key=lambda c: c.get("timestamp", 0), reverse=True)
            return checkpoints
        except Exception as e:
            logger.warning(f"Failed to list checkpoints: {e}")
            return []

    def load_named_checkpoint(self, name: str, mode: str = "watch") -> dict:
        """Load a named checkpoint. mode='watch' or 'train'."""
        if not self._redis:
            return {"error": "Redis not available"}

        data = self._redis.get(f"{self._checkpoint_key_prefix}{name}")
        if not data:
            return {"error": f"Checkpoint '{name}' not found"}

        try:
            # Pause training loop before modifying state
            self._training_active = False
            self._watch_mode = False
            # Wait for the training loop to reach its top (safe point)
            # Once _training_active=False, the loop can't reach _loop_idle.clear()
            # because the `if not self._training_active: continue` check fires first
            if not self._loop_idle.wait(timeout=5.0):
                logger.warning("Timed out waiting for training loop idle — proceeding anyway")
            self._loop_idle.clear()  # reset for next use

            buf = io.BytesIO(data)
            ckpt = torch.load(buf, map_location=self.device, weights_only=False)

            # Check action/obs size compatibility
            saved_action_size = ckpt.get("action_size", 3)
            if saved_action_size != self.action_size:
                self._training_active = True
                return {"error": f"Checkpoint uses action_size={saved_action_size}, current is {self.action_size}. Reset training first."}
            saved_obs_size = ckpt.get("obs_size", 30)
            if saved_obs_size != self.obs_size:
                self._training_active = True
                return {"error": f"Checkpoint uses obs_size={saved_obs_size}, current is {self.obs_size}. Reset training first."}

            if self.scenario == "soccer":
                saved_ppt = ckpt.get("players_per_team", self.env.players_per_team)
                if saved_ppt != self.env.players_per_team:
                    self.env.set_players_per_team(saved_ppt)

            try:
                self.model.load_state_dict(ckpt["model_state_dict"])
            except RuntimeError as arch_err:
                self._training_active = True
                return {"error": f"Checkpoint architecture mismatch (network resized?): {arch_err}"}
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.generation = ckpt.get("generation", 0)
            self.total_steps = ckpt.get("total_steps", 0)
            self.episode_count = ckpt.get("episode_count", 0)
            self._display_episode_count = ckpt.get("display_episode_count", 0)

            # Restore curriculum level (drone scenario)
            if self.scenario == "drone":
                saved_level = ckpt.get("curriculum_level", 1)
                self.env.set_curriculum_level(saved_level)
                self.n_steps = self.env.max_steps
                logger.info(f"[drone] Restored curriculum level {saved_level}")

            self._episode_rewards_cyan.clear()
            self._episode_rewards_magenta.clear()
            self._episode_goals.clear()
            self._episode_winners.clear()
            self._episode_rewards.clear()
            self._episode_gates.clear()
            self._episode_completions.clear()
            self._env_reward_acc_cyan[:] = 0.0
            self._env_reward_acc_magenta[:] = 0.0
            self._env_reward_acc[:] = 0.0
            self._total_goals_cyan = 0
            self._total_goals_magenta = 0
            self._stats = {}
            self.env.reset()

            self._loaded_checkpoint_name = name

            if mode == "watch":
                self.set_watch_mode()
            else:
                self.set_train_mode()

            logger.info(f"Loaded [{self.scenario}] checkpoint '{name}' in {mode} mode (gen={self.generation})")
            return {
                "status": "loaded",
                "name": name,
                "mode": mode,
                "generation": self.generation,
                "scenario": self.scenario,
            }
        except Exception as e:
            logger.warning(f"Failed to load checkpoint '{name}': {e}")
            return {"error": str(e)}

    def delete_checkpoint(self, name: str) -> dict:
        """Delete a named checkpoint."""
        if not self._redis:
            return {"error": "Redis not available"}
        try:
            self._redis.delete(f"{self._checkpoint_key_prefix}{name}")
            self._redis.hdel(self._checkpoint_index_key, name)
            if self._loaded_checkpoint_name == name:
                self._loaded_checkpoint_name = None
            logger.info(f"Deleted checkpoint: {name}")
            return {"status": "deleted", "name": name}
        except Exception as e:
            return {"error": str(e)}

    def stop(self):
        """Signal the training thread to stop."""
        self._running = False
        self._save_checkpoint()

    def _save_checkpoint(self):
        """Save model to Redis."""
        if not self._redis:
            return
        try:
            buf = io.BytesIO()
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "generation": self.generation,
                "total_steps": self.total_steps,
                "episode_count": self.episode_count,
                "display_episode_count": self._display_episode_count,
                "players_per_team": getattr(self.env, 'players_per_team', 1),
                "action_size": self.action_size,
                "obs_size": self.obs_size,
                "scenario": self.scenario,
                "curriculum_level": getattr(self.env, 'curriculum_level', 1),
            }, buf)
            self._redis.set(self._checkpoint_latest_key, buf.getvalue(), ex=86400)
            logger.info(f"[{self.scenario}] Checkpoint saved: gen={self.generation}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self):
        """Load model from Redis."""
        if not self._redis:
            return
        try:
            data = self._redis.get(self._checkpoint_latest_key)
            if data:
                buf = io.BytesIO(data)
                ckpt = torch.load(buf, map_location=self.device, weights_only=False)
                # Check action/obs size compatibility
                saved_action_size = ckpt.get("action_size", self.action_size)
                if saved_action_size != self.action_size:
                    logger.warning(
                        f"Checkpoint action size mismatch ({saved_action_size} vs {self.action_size}), "
                        f"starting fresh — reset training to use new action space"
                    )
                    return
                saved_obs_size = ckpt.get("obs_size", self.obs_size)
                if saved_obs_size != self.obs_size:
                    logger.warning(
                        f"Checkpoint obs size mismatch ({saved_obs_size} vs {self.obs_size}), "
                        f"starting fresh — reset training to use new observation space"
                    )
                    return
                # Restore team size before loading model (soccer only)
                if self.scenario == "soccer":
                    saved_ppt = ckpt.get("players_per_team", self.env.players_per_team)
                    if saved_ppt != self.env.players_per_team:
                        self.env.set_players_per_team(saved_ppt)
                try:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                except RuntimeError as arch_err:
                    logger.warning(
                        f"Checkpoint architecture mismatch (network resized?), starting fresh: {arch_err}"
                    )
                    return
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                self.generation = ckpt.get("generation", 0)
                self.total_steps = ckpt.get("total_steps", 0)
                self.episode_count = ckpt.get("episode_count", 0)
                self._display_episode_count = ckpt.get("display_episode_count", 0)
                # Restore curriculum level (drone scenario)
                if self.scenario == "drone":
                    saved_level = ckpt.get("curriculum_level", 1)
                    self.env.set_curriculum_level(saved_level)
                    self.n_steps = self.env.max_steps
                if self.scenario == "soccer":
                    logger.info(f"Checkpoint loaded: gen={self.generation}, {self.env.players_per_team}v{self.env.players_per_team}")
                else:
                    level_info = f", level={getattr(self.env, 'curriculum_level', '?')}" if self.scenario == "drone" else ""
                    logger.info(f"Checkpoint loaded [{self.scenario}]: gen={self.generation}, steps={self.total_steps}{level_info}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
