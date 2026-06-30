"""
firewall_env.py — Gymnasium environment for the RL-based firewall.

TWO ENVIRONMENTS ARE EXPORTED
------------------------------
  FlowFirewallEnv   (PRIMARY — flow-level episodes, gamma=0.99 justified)
  DynamicFirewallEnv (LEGACY ALIAS — per-packet bandit, kept for back-compat)

FlowFirewallEnv
---------------
Each episode = one synthetic flow (a sequence of T packets from the same
connection / attack family).  The LSTM agent accumulates hidden state across
steps, enabling temporal credit assignment:

  - s_{t+1} is correlated with s_t  (same flow → gamma=0.99 meaningful)
  - The agent can learn "block early" vs "gather evidence then block"
  - Rate-limit mid-flow then escalate to block is a learnable strategy

Reward shaping (temporal):
  - Blocking an attack *early*  (t < T/2):  full block_attack_reward
  - Blocking an attack *late*   (t >= T/2): 70 % of block_attack_reward
  - FN at the terminal packet:              false_negative_penalty  (worst)
  - FN at a non-terminal packet:            50 % of false_negative_penalty
    (partial: agent still has future steps to recover)
  - Rate-limit on attack:                   partial credit, linearly decaying
  - Correctly allowing benign:              benign_reward
  - FP (blocking benign):                   false_positive_penalty

All rewards are clamped to [-reward_clip, reward_clip].
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ── FlowFirewallEnv ───────────────────────────────────────────────────────────

class FlowFirewallEnv(gym.Env):
    """
    Flow-level Gymnasium environment for the LSTM-DQN firewall.

    Each call to reset() samples one synthetic flow episode:
        obs shape:    (num_features,)  — single-packet features
        action space: Discrete(3)      — 0=allow, 1=block, 2=rate-limit

    The LSTM hidden state in the agent carries temporal context across steps;
    the env itself is Markov in (packet_features, step_index) — the agent's
    recurrent memory provides the non-Markov part.

    Parameters
    ----------
    episodes : list[dict]
        Output of flow_builder.build_flow_episodes().  Each element is a dict
        with keys: features [T, F], labels [T], flow_id, attack_type, seq_len.
    normalise : bool
        If True, fit z-score normalisation stats on the concatenated feature
        matrix of all episodes at construction time.
    seed : int | None
        RNG seed for episode sampling.
    **reward_kwargs
        Override any reward parameter (see __init__ signature).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        episodes: List[Dict],
        # ── reward knobs ───────────────────────────────────────────────────
        benign_reward:           float = 1.0,
        block_attack_reward:     float = 8.0,
        false_positive_penalty:  float = -5.0,
        false_negative_penalty:  float = -10.0,
        rate_limit_penalty:      float = -0.3,
        reward_clip:             float = 10.0,
        # ── normalisation ──────────────────────────────────────────────────
        normalise:               bool  = True,
        seed:                    Optional[int] = None,
    ):
        super().__init__()

        if not episodes:
            raise ValueError("episodes list is empty")

        self.episodes: List[Dict] = episodes
        self.num_features: int    = episodes[0]["features"].shape[1]
        self._flow_len: int       = episodes[0]["features"].shape[0]

        # ── Normalisation ─────────────────────────────────────────────────
        # Fit mean/std on ALL features across ALL episodes (already done
        # during run_training before this env is created; here we accept
        # pre-normalised episodes, but store identity transforms as fallback).
        self.normalise = normalise
        if normalise:
            all_feats = np.vstack([ep["features"] for ep in episodes])
            self._mean = all_feats.mean(axis=0).astype(np.float32)
            self._std  = all_feats.std(axis=0).astype(np.float32)
            self._std[self._std < 1e-8] = 1.0
            # Normalise all episode feature matrices in place
            for ep in self.episodes:
                ep["features"] = (ep["features"] - self._mean) / self._std
        else:
            self._mean = np.zeros(self.num_features, dtype=np.float32)
            self._std  = np.ones(self.num_features,  dtype=np.float32)

        # ── Spaces ────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_features,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)   # 0=allow, 1=block, 2=rate-limit

        # ── Reward parameters ─────────────────────────────────────────────
        self.benign_reward          = benign_reward
        self.block_attack_reward    = block_attack_reward
        self.false_positive_penalty = false_positive_penalty
        self.false_negative_penalty = false_negative_penalty
        self.rate_limit_penalty     = rate_limit_penalty
        self.reward_clip            = reward_clip

        # ── Class-imbalance scaling (same logic as before) ────────────────
        n_attacks = sum(int(np.any(ep["labels"] == 1)) for ep in episodes)
        n_benign  = len(episodes) - n_attacks
        attack_ratio = n_attacks / max(n_benign, 1)
        if attack_ratio < 0.3:
            scale = min(3.0, 0.3 / max(attack_ratio, 1e-4))
            self.false_negative_penalty *= scale

        # ── Episode state ─────────────────────────────────────────────────
        self._current_episode: Optional[Dict] = None
        self._step_idx:        int             = 0
        self._flow_T:          int             = self._flow_len

        self.np_random, _ = gym.utils.seeding.np_random(seed)

    # ── Normalisation helper (used by run_firewall for live packets) ──────────
    def normalise_sample(self, x: np.ndarray) -> np.ndarray:
        return ((x.astype(np.float32) - self._mean) / self._std)

    # ── Reward function (temporal-aware) ─────────────────────────────────────
    def _compute_reward(self, action: int, label: int, step: int, total_steps: int) -> float:
        """
        Temporal reward shaping:

        Blocking attacks early (step < total_steps/2) gets FULL credit.
        Blocking attacks late  (step >= total_steps/2) gets 70% credit.
        False negatives at the *terminal* packet get the full penalty.
        False negatives at earlier packets get 50% penalty (agent can still
        recover in subsequent steps — this is the key temporal difference
        from the per-packet bandit, where every FN was equally penalised).

        Rate-limit: partial credit that decays as we get closer to terminal
        (the attacker has had more time, so rate-limiting becomes less valuable).
        """
        is_terminal  = (step == total_steps - 1)
        early        = step < (total_steps / 2)
        decay        = 1.0 - (step / max(total_steps - 1, 1)) * 0.3  # [1.0 → 0.7]

        if label == 0:   # benign
            if action == 0:
                reward = self.benign_reward
            elif action == 1:
                reward = self.false_positive_penalty
            else:        # rate-limit on benign: degraded service
                reward = self.benign_reward + self.rate_limit_penalty

        else:            # attack
            if action == 1:   # block
                reward = self.block_attack_reward if early else self.block_attack_reward * 0.7
            elif action == 0: # allow (false negative)
                if is_terminal:
                    reward = self.false_negative_penalty           # worst: missed entirely
                else:
                    reward = self.false_negative_penalty * 0.5    # partial: can still block
            else:             # rate-limit: partial credit, decays toward terminal
                base   = self.block_attack_reward * 0.5 * decay
                reward = base + self.rate_limit_penalty

        return float(np.clip(reward, -self.reward_clip, self.reward_clip))

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed:    Optional[int]  = None,
        options: Optional[Dict] = None,
    ):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Sample a random episode
        ep_idx = int(self.np_random.integers(0, len(self.episodes)))
        self._current_episode = self.episodes[ep_idx]
        self._step_idx        = 0
        self._flow_T          = self._current_episode["seq_len"]   # un-padded length

        obs  = self._current_episode["features"][0]
        info = {
            "flow_id":     self._current_episode["flow_id"],
            "attack_type": self._current_episode["attack_type"],
            "flow_len":    self._flow_T,
        }
        return obs.astype(np.float32), info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        assert self._current_episode is not None, "Call reset() before step()"

        ep     = self._current_episode
        t      = self._step_idx
        label  = int(ep["labels"][t])
        obs    = ep["features"][t]
        reward = self._compute_reward(action, label, t, self._flow_T)

        self._step_idx += 1
        # Episode terminates at the end of the actual (un-padded) flow
        terminated = (self._step_idx >= self._flow_T)
        truncated  = False

        if terminated:
            next_obs = np.zeros(self.num_features, dtype=np.float32)
        else:
            next_obs = ep["features"][self._step_idx].astype(np.float32)

        info = {
            "label":       label,
            "action":      action,
            "reward":      reward,
            "step":        t,
            "flow_len":    self._flow_T,
            "is_terminal": terminated,
            "flow_id":     ep["flow_id"],
            "attack_type": ep["attack_type"],
        }
        return next_obs, reward, terminated, truncated, info

    def render(self):
        if self._current_episode is not None:
            ep = self._current_episode
            print(
                f"[FlowFirewallEnv] flow={ep['flow_id']}  "
                f"step={self._step_idx}/{self._flow_T}  "
                f"type={ep['attack_type']}"
            )


# ── Legacy per-packet environment (kept for backwards-compat) ─────────────────

class DynamicFirewallEnv(gym.Env):
    """
    LEGACY per-packet contextual-bandit environment.

    Kept for backwards compatibility and ablation comparison.
    For new training, use FlowFirewallEnv instead.

    Note: gamma MUST remain 0.0 here — see agent_dqn.py DQNConfig docstring.
    The action taken on packet t has NO influence on which packet appears at
    t+1 (shuffled dataset traversal), making this a contextual bandit.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        traffic_samples: np.ndarray,
        labels:          np.ndarray,
        benign_reward:          float = 1.0,
        block_attack_reward:    float = 8.0,
        false_positive_penalty: float = -5.0,
        false_negative_penalty: float = -10.0,
        rate_limit_penalty:     float = -0.3,
        reward_clip:            float = 10.0,
        normalise:              bool  = True,
        seed:                   Optional[int] = None,
    ):
        super().__init__()

        assert traffic_samples.shape[0] == labels.shape[0]

        self.traffic_samples = traffic_samples.astype(np.float32)
        self.labels          = labels.astype(np.int32)
        self.num_samples, self.num_features = self.traffic_samples.shape

        self.normalise = normalise
        if normalise:
            self._mean = self.traffic_samples.mean(axis=0)
            self._std  = self.traffic_samples.std(axis=0)
            self._std[self._std < 1e-8] = 1.0
            self.traffic_samples = (self.traffic_samples - self._mean) / self._std
        else:
            self._mean = np.zeros(self.num_features, dtype=np.float32)
            self._std  = np.ones(self.num_features,  dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_features,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self.benign_reward          = benign_reward
        self.block_attack_reward    = block_attack_reward
        self.false_positive_penalty = false_positive_penalty
        self.false_negative_penalty = false_negative_penalty
        self.rate_limit_penalty     = rate_limit_penalty
        self.reward_clip            = reward_clip

        n_attacks    = int(np.sum(labels == 1))
        n_benign     = int(np.sum(labels == 0))
        attack_ratio = n_attacks / max(n_benign, 1)
        if attack_ratio < 0.3:
            scale = min(3.0, 0.3 / max(attack_ratio, 1e-4))
            self.false_negative_penalty *= scale

        self._benign_idx  = np.where(self.labels == 0)[0]
        self._attack_idx  = np.where(self.labels == 1)[0]
        self.current_index   = 0
        self._episode_order: np.ndarray = np.arange(self.num_samples)
        self.np_random, _    = gym.utils.seeding.np_random(seed)

    def normalise_sample(self, x: np.ndarray) -> np.ndarray:
        return ((x.astype(np.float32) - self._mean) / self._std)

    def _compute_reward(self, action: int, label: int) -> float:
        if label == 0:
            if action == 0:   reward = self.benign_reward
            elif action == 1: reward = self.false_positive_penalty
            else:             reward = self.benign_reward + self.rate_limit_penalty
        else:
            if action == 1:   reward = self.block_attack_reward
            elif action == 0: reward = self.false_negative_penalty
            else:             reward = self.block_attack_reward * 0.6 + self.rate_limit_penalty
        return float(np.clip(reward, -self.reward_clip, self.reward_clip))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        b_idx = self.np_random.permutation(self._benign_idx)
        a_idx = self.np_random.permutation(self._attack_idx)

        all_idx = np.empty(self.num_samples, dtype=np.int64)
        ratio   = len(b_idx) / max(len(a_idx), 1)
        bi = ai = pos = 0
        while bi < len(b_idx) or ai < len(a_idx):
            n_b = int(ratio) if ai < len(a_idx) else len(b_idx) - bi
            for _ in range(n_b):
                if bi < len(b_idx):
                    all_idx[pos] = b_idx[bi]; bi += 1; pos += 1
            if ai < len(a_idx):
                all_idx[pos] = a_idx[ai]; ai += 1; pos += 1

        self._episode_order = all_idx[:pos]
        self.current_index  = 0
        obs = self.traffic_samples[self._episode_order[self.current_index]]
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        idx    = self._episode_order[self.current_index]
        label  = int(self.labels[idx])
        obs    = self.traffic_samples[idx]
        reward = self._compute_reward(action, label)

        self.current_index += 1
        terminated = self.current_index >= len(self._episode_order)
        truncated  = False

        if terminated:
            next_obs = np.zeros(self.num_features, dtype=np.float32)
        else:
            next_obs = self.traffic_samples[self._episode_order[self.current_index]]

        info = {"label": label, "action": action, "reward": reward, "sample_idx": int(idx)}
        return next_obs, reward, terminated, truncated, info

    def render(self):
        pass
