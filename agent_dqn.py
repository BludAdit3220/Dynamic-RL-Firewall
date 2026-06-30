"""
agent_dqn.py — LSTM Double DQN agent for the flow-level RL firewall.

ARCHITECTURE OVERVIEW
---------------------
The core change from the per-packet MLP agent is the introduction of an LSTM
Q-network that carries hidden state (h_t, c_t) across steps within a flow
episode.  This is the mechanism that makes gamma=0.99 genuinely useful:

    obs[t] → LinearEncoder → LSTM(h_{t-1}, c_{t-1}) → DuelingHead → Q(s,a)

The LSTM hidden state *is* the agent's intra-flow memory.  After seeing
packets 0..t of a flow, the hidden state encodes the trajectory so far,
letting the Q-values for packet t+1 depend on the entire flow history.

REPLAY BUFFER (EPISODE-LEVEL)
------------------------------
Flat transition buffers ((s,a,r,s',done) tuples) don't work for LSTM
training because independent transitions give the LSTM no useful sequence
context.  Instead, we store **complete flow episodes** and sample fixed-length
sub-sequences (length=seq_len) for BPTT (backpropagation through time).

Each sampled batch is shaped [B, T, F] rather than [B, F], and the LSTM
is unrolled for T steps per sample.  The hidden state at the start of each
BPTT window is initialised to zeros (simple burn-in approximation — valid for
short flows of ≤ 20 steps).

DOUBLE DQN + DUELING NETWORK
------------------------------
- Double DQN: action *selection* uses the online network; action *evaluation*
  uses the target network.  This removes maximisation bias.
- Dueling streams: the LSTM output is split into a value stream V(s) and an
  advantage stream A(s,a), recombined as Q(s,a) = V(s) + A(s,a) - mean(A).
  This improves learning when the "which action" decision matters less than
  "how good is this state" (common in early flow steps where the flow is
  ambiguous).
- gamma defaults to 0.99 — justified because consecutive steps in the same
  flow episode are causally correlated.

CHANGES FROM THE PER-PACKET AGENT
----------------------------------
  1. QNetwork   → LSTMQNetwork  (LSTM + dueling head)
  2. ReplayBuffer → EpisodeReplayBuffer  (stores episodes, samples sequences)
  3. DQNAgent  → LSTMDQNAgent
     - select_action(obs, h, c) returns (action, h_new, c_new)
     - store_episode(episode_transitions) adds a full episode
     - train_step() samples sequences and runs BPTT
     - reset_hidden() / init_hidden() manage episode boundaries
  4. DQNConfig: gamma=0.99 (was 0.0), seq_len=8 (BPTT window)

BACKWARDS-COMPAT ALIASES
-------------------------
DQNAgent and DQNConfig still exist as aliases to the LSTM variants so that
existing scripts that import them continue to work.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LSTMDQNConfig:
    state_dim:            int
    num_actions:          int   = 3

    # Flow-level RL: consecutive steps in the same flow are causally related,
    # so gamma=0.99 is justified (unlike the per-packet bandit where it was 0).
    gamma:                float = 0.99

    learning_rate:        float = 3e-4
    batch_size:           int   = 32          # number of sequences per batch
    seq_len:              int   = 8           # BPTT unroll length
    buffer_capacity:      int   = 10_000      # max episodes stored
    min_episodes_before_train: int = 200      # warm-up

    # LSTM architecture
    encoder_dim:          int   = 128         # Linear encoder width
    lstm_hidden:          int   = 256         # LSTM hidden size
    dueling:              bool  = True        # dueling value/advantage heads

    # Target network
    tau:                  float = 0.005       # soft-update coefficient

    # Exploration
    epsilon_start:        float = 1.0
    epsilon_end:          float = 0.05
    epsilon_decay_steps:  int   = 200_000

    # Stability
    grad_clip:            float = 10.0
    q_value_clip:         float = 50.0
    minority_oversample:  int   = 2           # attack episodes stored N times


# Back-compat alias
DQNConfig = LSTMDQNConfig


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Q-Network (Dueling)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMQNetwork(nn.Module):
    """
    Per-step Q-network with an LSTM core.

    Forward signature for *training* (batched sequences):
        forward(x, h, c)  where x: [B, T, F]
        returns (q_values: [B, T, num_actions], h_new, c_new)

    Forward signature for *inference* (single step):
        forward_step(x, h, c)  where x: [1, F]
        returns (q_values: [1, num_actions], h_new, c_new)

    Architecture:
        x[t] → Linear(F, encoder_dim) → LayerNorm → ReLU
             → LSTM(encoder_dim, lstm_hidden)
             → Dueling head: Linear → [V(s), A(s,a)]
             → Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """

    def __init__(self, cfg: LSTMDQNConfig):
        super().__init__()
        self.cfg = cfg

        # Packet encoder: maps raw features to LSTM input
        self.encoder = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.encoder_dim),
            nn.LayerNorm(cfg.encoder_dim),
            nn.ReLU(),
        )

        # LSTM carries temporal context across flow steps
        self.lstm = nn.LSTM(
            input_size=cfg.encoder_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=1,
            batch_first=True,   # input: [batch, seq, features]
        )

        if cfg.dueling:
            # Value stream: how good is this flow state?
            self.value_stream = nn.Sequential(
                nn.Linear(cfg.lstm_hidden, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            # Advantage stream: which action is best given this state?
            self.advantage_stream = nn.Sequential(
                nn.Linear(cfg.lstm_hidden, 128),
                nn.ReLU(),
                nn.Linear(128, cfg.num_actions),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(cfg.lstm_hidden, 128),
                nn.ReLU(),
                nn.Linear(128, cfg.num_actions),
            )

    def forward(
        self,
        x:   torch.Tensor,              # [B, T, F]
        h:   Optional[torch.Tensor],    # [1, B, lstm_hidden] or None
        c:   Optional[torch.Tensor],    # [1, B, lstm_hidden] or None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched sequence forward pass for training."""
        B, T, F = x.shape

        # Encode each packet independently
        x_flat   = x.reshape(B * T, F)
        enc_flat = self.encoder(x_flat)
        enc      = enc_flat.reshape(B, T, -1)   # [B, T, encoder_dim]

        # LSTM unroll
        if h is None or c is None:
            lstm_out, (h_new, c_new) = self.lstm(enc)
        else:
            lstm_out, (h_new, c_new) = self.lstm(enc, (h, c))
        # lstm_out: [B, T, lstm_hidden]

        q = self._q_from_lstm(lstm_out)   # [B, T, num_actions]
        return q, h_new, c_new

    def forward_step(
        self,
        x: torch.Tensor,                # [1, F]  (single packet, no seq dim)
        h: Optional[torch.Tensor],
        c: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step forward pass for inference (no BPTT)."""
        enc = self.encoder(x)               # [1, encoder_dim]
        enc = enc.unsqueeze(1)              # [1, 1, encoder_dim] (add seq dim)

        if h is None or c is None:
            lstm_out, (h_new, c_new) = self.lstm(enc)
        else:
            lstm_out, (h_new, c_new) = self.lstm(enc, (h, c))

        lstm_out = lstm_out.squeeze(1)      # [1, lstm_hidden]
        q = self._q_from_lstm_step(lstm_out)   # [1, num_actions]
        return q, h_new, c_new

    def _q_from_lstm(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Compute Q-values from LSTM output [B, T, hidden]."""
        if self.cfg.dueling:
            V = self.value_stream(lstm_out)         # [B, T, 1]
            A = self.advantage_stream(lstm_out)     # [B, T, num_actions]
            return V + (A - A.mean(dim=-1, keepdim=True))
        else:
            return self.q_head(lstm_out)

    def _q_from_lstm_step(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Compute Q-values from LSTM output [B, hidden] (no time dim)."""
        if self.cfg.dueling:
            V = self.value_stream(lstm_out)         # [B, 1]
            A = self.advantage_stream(lstm_out)     # [B, num_actions]
            return V + (A - A.mean(dim=-1, keepdim=True))
        else:
            return self.q_head(lstm_out)

    def init_hidden(self, batch_size: int, device: torch.device):
        """Return zero hidden state (h_0, c_0) for a new episode."""
        h = torch.zeros(1, batch_size, self.cfg.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.cfg.lstm_hidden, device=device)
        return h, c


# ─────────────────────────────────────────────────────────────────────────────
# Episode Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeTransition:
    """One step within a stored episode."""
    state:      np.ndarray  # [F]
    action:     int
    reward:     float
    next_state: np.ndarray  # [F]
    done:       bool
    label:      int


class EpisodeReplayBuffer:
    """
    Replay buffer that stores complete flow *episodes* rather than flat
    (s, a, r, s', done) transitions.

    At sample time, a batch of B episodes is drawn; from each episode a
    random sub-sequence of length seq_len is extracted.  This gives the LSTM
    exactly seq_len steps of causal context without having to store or
    reconstruct hidden states.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[List[EpisodeTransition]] = deque(maxlen=capacity)

    def add_episode(self, transitions: List[EpisodeTransition]):
        """Store one complete episode."""
        if len(transitions) > 0:
            self.buffer.append(transitions)

    def sample(
        self,
        batch_size: int,
        seq_len:    int,
    ) -> Tuple[np.ndarray, ...]:
        """
        Sample batch_size sub-sequences of length seq_len.

        Returns arrays of shape:
            states:      [B, T, F]
            actions:     [B, T]
            rewards:     [B, T]
            next_states: [B, T, F]
            dones:       [B, T]
        where T = seq_len.
        """
        episodes = random.choices(list(self.buffer), k=batch_size)

        states_list      = []
        actions_list     = []
        rewards_list     = []
        next_states_list = []
        dones_list       = []

        for ep in episodes:
            T    = len(ep)
            # Pick a random starting point so the full seq_len fits
            start = random.randint(0, max(0, T - seq_len))
            chunk = ep[start : start + seq_len]

            # Pad with copies of the last transition if episode is too short
            while len(chunk) < seq_len:
                chunk.append(chunk[-1])

            states_list.append([t.state      for t in chunk])
            actions_list.append([t.action    for t in chunk])
            rewards_list.append([t.reward    for t in chunk])
            next_states_list.append([t.next_state for t in chunk])
            dones_list.append([float(t.done) for t in chunk])

        return (
            np.array(states_list,      dtype=np.float32),   # [B, T, F]
            np.array(actions_list,     dtype=np.int64),     # [B, T]
            np.array(rewards_list,     dtype=np.float32),   # [B, T]
            np.array(next_states_list, dtype=np.float32),   # [B, T, F]
            np.array(dones_list,       dtype=np.float32),   # [B, T]
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM DQN Agent
# ─────────────────────────────────────────────────────────────────────────────

class LSTMDQNAgent:
    """
    Double DQN agent with LSTM Q-network for flow-level sequential decisions.

    Key API differences from the flat DQNAgent:
      - select_action(state, h, c, training)  → (action, h_new, c_new)
      - store_episode(episode_transitions)     → adds a full episode
      - init_hidden()                          → zero (h, c) for new episode
      - train_step()                           → samples sequences, runs BPTT
    """

    def __init__(self, config: LSTMDQNConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_network = LSTMQNetwork(config).to(self.device)
        self.target_network = LSTMQNetwork(config).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer  = optim.Adam(self.online_network.parameters(), lr=config.learning_rate)
        self.loss_fn    = nn.SmoothL1Loss()

        self.replay_buffer = EpisodeReplayBuffer(config.buffer_capacity)
        self.epsilon        = config.epsilon_start
        self.global_step    = 0

    # ── Hidden-state helpers ───────────────────────────────────────────────

    def init_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero (h, c) for the start of a new episode."""
        return self.online_network.init_hidden(1, self.device)

    # ── Action selection ───────────────────────────────────────────────────

    def select_action(
        self,
        state:    np.ndarray,
        h:        torch.Tensor,
        c:        torch.Tensor,
        training: bool = True,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select an action for one step within a flow.

        Parameters
        ----------
        state   : [F] numpy array (single packet observation)
        h, c    : LSTM hidden state from the previous step (or init_hidden())
        training: if True, apply epsilon-greedy exploration

        Returns
        -------
        (action: int, h_new, c_new)  — pass h_new, c_new to the next step
        """
        if training and random.random() < self.epsilon:
            # Random action — but still advance hidden state so the LSTM
            # continues to accumulate context even during exploration.
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, h_new, c_new = self.online_network.forward_step(t, h, c)
            return random.randrange(self.config.num_actions), h_new, c_new

        with torch.no_grad():
            t          = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q, h_new, c_new = self.online_network.forward_step(t, h, c)
        return int(q.argmax(dim=1).item()), h_new, c_new

    # ── Epsilon schedule ───────────────────────────────────────────────────

    def _update_epsilon(self):
        frac = min(1.0, self.global_step / self.config.epsilon_decay_steps)
        self.epsilon = (
            self.config.epsilon_start
            + frac * (self.config.epsilon_end - self.config.epsilon_start)
        )

    # ── Episode storage ────────────────────────────────────────────────────

    def store_episode(
        self,
        transitions: List[EpisodeTransition],
        is_attack:   bool = False,
    ):
        """
        Store one complete episode in the replay buffer.

        Attack episodes are stored minority_oversample times to counteract
        the class imbalance in the episode pool.
        """
        repeat = self.config.minority_oversample if is_attack else 1
        for _ in range(repeat):
            self.replay_buffer.add_episode(transitions)

    # ── Training step (BPTT over sampled sequences) ────────────────────────

    def train_step(self) -> Optional[float]:
        """
        Sample a batch of sub-sequences and run one Double-DQN update via BPTT.

        Returns the scalar loss, or None if the buffer isn't warm yet.
        """
        if len(self.replay_buffer) < self.config.min_episodes_before_train:
            return None

        cfg = self.config
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            cfg.batch_size, cfg.seq_len
        )

        B, T, F = states.shape

        s  = torch.FloatTensor(states).to(self.device)       # [B, T, F]
        a  = torch.LongTensor(actions).to(self.device)       # [B, T]
        r  = torch.FloatTensor(rewards).to(self.device)      # [B, T]
        s2 = torch.FloatTensor(next_states).to(self.device)  # [B, T, F]
        d  = torch.FloatTensor(dones).to(self.device)        # [B, T]

        # Zero hidden states for BPTT window (burn-in approximation)
        h0_online = torch.zeros(1, B, cfg.lstm_hidden, device=self.device)
        c0_online = torch.zeros(1, B, cfg.lstm_hidden, device=self.device)
        h0_target = torch.zeros(1, B, cfg.lstm_hidden, device=self.device)
        c0_target = torch.zeros(1, B, cfg.lstm_hidden, device=self.device)

        # ── Double DQN targets ──────────────────────────────────────────────
        with torch.no_grad():
            # Online network selects best next action
            q_online_next, _, _ = self.online_network(s2, h0_online, c0_online)
            best_next_actions = q_online_next.argmax(dim=-1, keepdim=True)  # [B, T, 1]

            # Target network evaluates that action's Q-value
            q_target_next, _, _ = self.target_network(s2, h0_target, c0_target)
            next_q = q_target_next.gather(-1, best_next_actions).squeeze(-1)  # [B, T]

            targets = r + cfg.gamma * next_q * (1.0 - d)
            targets = torch.clamp(targets, -cfg.q_value_clip, cfg.q_value_clip)

        # ── Current Q-values ────────────────────────────────────────────────
        q_values, _, _ = self.online_network(s, h0_online, c0_online)  # [B, T, A]
        current_q = q_values.gather(-1, a.unsqueeze(-1)).squeeze(-1)    # [B, T]

        loss = self.loss_fn(current_q, targets)

        if not torch.isfinite(loss):
            print("[agent_dqn] WARNING: non-finite loss — skipping update.")
            self.optimizer.zero_grad()
            return None

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), cfg.grad_clip)
        self.optimizer.step()

        self._soft_update_target()
        self.global_step += 1
        self._update_epsilon()

        return float(loss.item())

    # ── Soft target-network update ─────────────────────────────────────────

    def _soft_update_target(self):
        tau = self.config.tau
        for op, tp in zip(
            self.online_network.parameters(),
            self.target_network.parameters(),
        ):
            tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.online_network.state_dict(),
                "config":     self.config,
                "agent_type": "lstm",
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        sd   = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.online_network.load_state_dict(sd)
        self.target_network.load_state_dict(sd)
        self.target_network.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Back-compat flat replay buffer (used by legacy DynamicFirewallEnv path)
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Flat (s,a,r,s',done) replay buffer — legacy per-packet agent only."""
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            states.astype(np.float32),
            actions.astype(np.int64),
            rewards.astype(np.float32),
            next_states.astype(np.float32),
            dones.astype(np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# Back-compat alias so existing scripts that do
#   from agent_dqn import DQNAgent, DQNConfig
# continue to work.
DQNAgent = LSTMDQNAgent
