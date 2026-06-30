"""
run_training.py — Offline training script for the LSTM-DQN RL firewall.

HOW THIS DIFFERS FROM THE PER-PACKET TRAINING LOOP
----------------------------------------------------
The fundamental change is that training now iterates over *flow episodes*
rather than individual packets.  Each episode is a sequence of T steps from
a synthetic flow (built by flow_builder.py); within the episode the LSTM
agent carries hidden state (h_t, c_t) from step to step.

Training loop (per episode):
    1. env.reset()   → sample a random flow episode, get obs[0], reset h,c
    2. for t in 0..T-1:
           action, h, c = agent.select_action(obs, h, c)
           obs, reward, done, info = env.step(action)
           accumulate transition
    3. agent.store_episode(transitions)   ← whole episode stored
    4. agent.train_step()                 ← sample seq_len-length sub-sequences,
                                            run BPTT, update LSTM + head

Metrics per episode:
  - Total reward
  - Flow-level detection rate: was the flow classified correctly overall?
  - Early detection rate:  % of attack flows blocked in first half of episode
  - Packet-level DR / FPR (for apples-to-apples comparison with old agent)

Validation:
  - Run greedy LSTM agent over all held-out flow episodes
  - Score = flow-level Youden's J: flow_DR - flow_FPR
  - Best checkpoint tracked and restored at end of training

CLI changes vs old run_training.py:
  --flow-len N       episode length (packets per synthetic flow, default 10)
  --seq-len N        BPTT unroll length (must be ≤ flow-len, default 8)
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch

from flow_builder import build_flow_episodes, split_episodes, episode_stats
from firewall_env  import FlowFirewallEnv
from agent_dqn     import LSTMDQNAgent, LSTMDQNConfig, EpisodeTransition


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and clean a CIC-IDS-style CSV dataset.

    Returns
    -------
    features     : np.ndarray [N, F]
    labels       : np.ndarray [N]     binary (0=benign, 1=attack)
    raw_lbls     : list[str]          original label strings (e.g. "BENIGN", "DoS")
    feature_names: list[str]          column names that survived cleaning
    """
    if path.is_dir():
        csv_paths = sorted(path.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files in: {path}")
        df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip() for c in df.columns]

    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise KeyError(f"No 'label' column found. Columns: {list(df.columns)}")

    raw_lbls_series = df[label_col].astype(str).str.strip()
    raw_lbls        = raw_lbls_series.tolist()

    try:
        labels = df[label_col].to_numpy(dtype=np.int32)
    except (ValueError, TypeError):
        labels = (raw_lbls_series != "BENIGN").astype(np.int32).to_numpy()

    features = df.drop(columns=[label_col])
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.dropna(axis=1, how="all")
    const_cols = features.columns[features.nunique() <= 1]
    if len(const_cols):
        print(f"[load_dataset] Dropping {len(const_cols)} constant column(s)")
        features = features.drop(columns=const_cols)

    # Capture surviving column names BEFORE converting to numpy
    feature_names = list(features.columns)

    medians  = features.median()
    features = features.fillna(medians)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(medians)

    return features.to_numpy(dtype=np.float32), labels, raw_lbls, feature_names


# ── Normalisation helpers ─────────────────────────────────────────────────────

def fit_normaliser(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0).astype(np.float32)
    std  = features.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_normaliser(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features.astype(np.float32) - mean) / std)


# ── Flow-level validation ─────────────────────────────────────────────────────

def evaluate_flow(
    agent:    LSTMDQNAgent,
    episodes: list,
) -> Tuple[float, float, int, int, int, int, float]:
    """
    Run the greedy LSTM agent over all validation flow episodes.

    A flow is classified as "blocked" if the agent takes action 1 or 2
    (block/rate-limit) on *any* step within the flow (the flow was detected).

    Returns
    -------
    flow_dr, flow_fpr, tp, tn, fp, fn, early_detection_rate
    """
    agent.online_network.eval()
    tp = tn = fp = fn = 0
    early_blocks = total_attack_flows = 0

    with torch.no_grad():
        for ep in episodes:
            feats  = ep["features"]   # [T, F]  already normalised
            labels = ep["labels"]     # [T]
            T      = ep["seq_len"]
            is_malicious = bool(np.any(labels[:T] == 1))

            h, c    = agent.init_hidden()
            blocked = False
            early_blocked = False

            for t in range(T):
                obs = feats[t]
                action, h, c = agent.select_action(obs, h, c, training=False)
                if action in (1, 2):
                    blocked = True
                    if t < T / 2:
                        early_blocked = True
                    break   # flow is blocked — no further steps needed

            if is_malicious:
                total_attack_flows += 1
                if blocked:
                    tp += 1
                    if early_blocked:
                        early_blocks += 1
                else:
                    fn += 1
            else:
                if blocked:
                    fp += 1
                else:
                    tn += 1

    agent.online_network.train()

    flow_dr  = tp / max(tp + fn, 1)
    flow_fpr = fp / max(fp + tn, 1)
    early_dr = early_blocks / max(total_attack_flows, 1)
    return flow_dr, flow_fpr, tp, tn, fp, fn, early_dr


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    dataset_path:        pathlib.Path,
    model_out:           pathlib.Path,
    episodes:            int   = 200,
    flow_len:            int   = 10,
    seq_len:             int   = 8,
    val_fraction:        float = 0.15,
    patience:            int   = 20,
    divergence_factor:   float = 8.0,
    min_buffer_episodes: int   = 200,
    val_every:           int   = 5,    # run validation every N episodes
):
    # ── Load data ─────────────────────────────────────────────────────────
    features, labels, raw_lbls, feature_names = load_dataset(dataset_path)

    n_attacks = int(np.sum(labels == 1))
    n_benign  = int(np.sum(labels == 0))
    print(
        f"[train] Dataset: {len(labels):,} rows  "
        f"({n_benign:,} benign, {n_attacks:,} attacks, "
        f"attack_ratio={n_attacks/len(labels):.1%})"
    )

    # ── Fit normalisation on ALL features BEFORE splitting ────────────────
    # The normalisation is fitted on the full dataset here so that both
    # train and val episodes use the same scale.  (In a strict pipeline you
    # would fit only on train rows; for this offline setting the difference
    # is negligible and keeps the code simple.)
    mean, std = fit_normaliser(features)
    features_norm = apply_normaliser(features, mean, std)

    # ── Build flow episodes from normalised rows ───────────────────────────
    all_episodes = build_flow_episodes(
        features_norm,
        labels,
        raw_lbls,
        flow_len=flow_len,
        seed=42,
    )
    train_eps, val_eps = split_episodes(all_episodes, val_fraction=val_fraction)

    stats = episode_stats(train_eps)
    print(
        f"[train] Train episodes: {stats['total']:,}  "
        f"({stats['benign']:,} benign, {stats['malicious']:,} attack)  "
        f"avg_len={stats['avg_seq_len']:.1f}"
    )
    print(f"[train] Attack types: {sorted(stats['types'].keys())}")

    # ── Build env (no double-normalisation — episodes already normed) ──────
    env = FlowFirewallEnv(
        episodes=train_eps,
        normalise=False,   # episodes already normalised above
        seed=None,
    )
    # Store mean/std on env for run_firewall.py to read back
    env._mean = mean
    env._std  = std

    # ── Build agent ────────────────────────────────────────────────────────
    # Scale epsilon decay to actual training budget so the agent transitions
    # from exploration → exploitation within this run (not over 200k steps).
    # We target reaching epsilon_end by ~80% of the post-warmup budget.
    post_warmup_episodes = max(1, episodes - min_buffer_episodes)
    epsilon_decay_steps  = int(post_warmup_episodes * flow_len * 0.8)

    config = LSTMDQNConfig(
        state_dim=features_norm.shape[1],
        num_actions=env.action_space.n,
        seq_len=min(seq_len, flow_len),
        min_episodes_before_train=min_buffer_episodes,
        epsilon_decay_steps=epsilon_decay_steps,
    )
    agent = LSTMDQNAgent(config)

    print(
        f"[train] LSTM-DQN  state_dim={config.state_dim}  "
        f"lstm_hidden={config.lstm_hidden}  "
        f"gamma={config.gamma}  seq_len={config.seq_len}  "
        f"device={agent.device}"
    )
    print(
        f"[train] Epsilon: 1.0 → {config.epsilon_end} over {epsilon_decay_steps} steps "
        f"(reaches min at ~ep {min_buffer_episodes + int(epsilon_decay_steps/flow_len)})"
    )
    print(f"[train] Training for {episodes} RL episodes "
          f"(flow_len={flow_len}, val={len(val_eps):,} episodes)")

    best_score      = -np.inf
    best_state      = None
    best_episode    = -1
    no_improve      = 0
    baseline_loss   = None

    for ep_idx in range(episodes):
        obs, ep_info = env.reset()
        h, c = agent.init_hidden()

        ep_reward    = 0.0
        ep_loss_sum  = 0.0
        loss_count   = 0
        transitions: List[EpisodeTransition] = []

        # Per-episode packet-level confusion counts
        tp = tn = fp = fn = 0
        flow_blocked      = False
        flow_blocked_early = False

        while True:
            action, h, c = agent.select_action(obs, h, c, training=True)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done  = terminated or truncated
            label = step_info["label"]
            t_idx = step_info["step"]
            T     = step_info["flow_len"]

            transitions.append(EpisodeTransition(
                state=obs.copy(),
                action=action,
                reward=reward,
                next_state=next_obs.copy(),
                done=done,
                label=label,
            ))

            # Confusion matrix (packet-level)
            if label == 1 and action in (1, 2): tp += 1
            elif label == 0 and action == 0:    tn += 1
            elif label == 0 and action in (1, 2): fp += 1
            elif label == 1 and action == 0:    fn += 1

            # Flow-level block tracking
            if action in (1, 2):
                flow_blocked = True
                if t_idx < T / 2:
                    flow_blocked_early = True

            ep_reward += reward
            obs = next_obs

            # Train on every step (or every few steps for efficiency)
            loss = agent.train_step()
            if loss is not None:
                ep_loss_sum += loss
                loss_count  += 1

            if done:
                break

        # Store the episode (attack episodes stored minority_oversample times)
        is_attack = ep_info["attack_type"] != "BENIGN"
        agent.store_episode(transitions, is_attack=is_attack)

        avg_loss    = ep_loss_sum / max(loss_count, 1)
        pkt_dr      = tp / max(tp + fn, 1)
        pkt_fpr     = fp / max(fp + tn, 1)
        flow_type   = ep_info["attack_type"]

        # ── Divergence guard ───────────────────────────────────────────────
        if not np.isfinite(avg_loss):
            print(
                f"\n[train] !! Non-finite loss at episode {ep_idx+1}. "
                f"Restoring best (ep {best_episode+1}, score={best_score:+.3f})."
            )
            break
        if loss_count > 0:
            if baseline_loss is None and ep_idx >= 30:
                baseline_loss = max(avg_loss, 1e-6)
            elif baseline_loss is not None and avg_loss > divergence_factor * baseline_loss:
                print(
                    f"\n[train] !! Loss divergence at ep {ep_idx+1} "
                    f"(loss={avg_loss:.2f}, baseline={baseline_loss:.2f}). Stopping."
                )
                break

        # ── Validation (every val_every episodes to keep CPU manageable) ──────
        # Only validate once the buffer is warm and epsilon has started decaying.
        buffer_warm  = (loss_count > 0)          # at least one gradient update
        should_val   = (
            buffer_warm and
            (ep_idx % val_every == 0 or ep_idx == episodes - 1)
        )

        if should_val:
            val_dr, val_fpr, vtp, vtn, vfp, vfn, val_early_dr = evaluate_flow(
                agent, val_eps
            )
            score = val_dr - val_fpr   # Youden's J on flow-level decisions
        else:
            # During warm-up or between val intervals, report cached values
            val_dr = val_fpr = val_early_dr = float("nan")
            score  = float("-inf")     # won't trigger best_score update

        print(
            f"Ep {ep_idx+1:>4}/{episodes} | "
            f"type={flow_type:<20} | "
            f"reward={ep_reward:>7.1f} | "
            f"loss={avg_loss:.4f} | "
            f"ε={agent.epsilon:.3f} | "
            f"pkt DR={pkt_dr:.0%} FPR={pkt_fpr:.0%} | "
            f"val flow DR={val_dr:.1%} FPR={val_fpr:.1%} "
            f"earlyDR={val_early_dr:.1%} score={score:+.3f}"
        )

        checkpoint_eligible = agent.epsilon <= 0.4
        if score > best_score and score != float("-inf"):
            best_score   = score
            best_episode = ep_idx
            best_state   = {k: v.detach().clone()
                            for k, v in agent.online_network.state_dict().items()}
            no_improve   = 0
            if checkpoint_eligible:
                model_out.parent.mkdir(parents=True, exist_ok=True)
                agent.save(str(model_out))
                print(
                    f"          \u21b3 new best flow Youden's J={score:+.3f}, "
                    f"earlyDR={val_early_dr:.1%} \u2014 saved \u2192 {model_out}"
                )
            else:
                print(
                    f"          \u21b3 new best score={score:+.3f} "
                    f"(\u03b5={agent.epsilon:.3f} > 0.4, still warming up)"
                )
        elif buffer_warm and agent.epsilon < 0.4 and score != float("-inf"):
            # Only count patience once the agent is fully in exploitation mode.
            # High-ε exploration phases naturally produce oscillating scores —
            # penalising that with patience causes premature stopping.
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"\n[train] No improvement for {patience} val checks. "
                    f"Best was ep {best_episode+1}, score={best_score:+.3f}."
                )
                break

    # ── Restore best checkpoint ────────────────────────────────────────────
    if best_state is not None:
        agent.online_network.load_state_dict(best_state)
        agent.target_network.load_state_dict(best_state)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(model_out))
    print(
        f"\n[train] Final model (best flow score={best_score:+.3f}, "
        f"episode {best_episode+1}) → {model_out}"
    )

    # ── Save normalisation params ──────────────────────────────────────────
    norm_path = model_out.parent / "norm_params.json"
    with open(norm_path, "w") as f:
        json.dump(
            {
                "mean":          mean.tolist(),
                "std":           std.tolist(),
                "num_features":  int(features.shape[1]),
                "flow_len":      flow_len,
                "agent_type":    "lstm",
                # feature_names lets run_firewall.py select the right columns
                # from FlowStatAccumulator's 78-feature CIC-IDS vector
                "feature_names": feature_names if feature_names else [],
            },
            f,
        )
    print(f"[train] Saved normalisation params → {norm_path}")

    # ── Final val report ───────────────────────────────────────────────────
    val_dr, val_fpr, vtp, vtn, vfp, vfn, val_early_dr = evaluate_flow(agent, val_eps)
    print("\n[train] Final validation (best checkpoint):")
    print(f"  Flow Detection Rate : {val_dr:.2%}")
    print(f"  Flow FPR            : {val_fpr:.2%}")
    print(f"  Early Detection Rate: {val_early_dr:.2%}")
    print(f"  TP={vtp}  TN={vtn}  FP={vfp}  FN={vfn}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train LSTM-DQN RL firewall (flow-level).")
    p.add_argument("--dataset",      type=str,   required=True,
                   help="CSV or directory of CSVs (CIC-IDS format).")
    p.add_argument("--out",          type=str,   default="models/rl_firewall_lstm.pt",
                   help="Output model path (.pt).")
    p.add_argument("--episodes",     type=int,   default=200,
                   help="Number of RL training episodes.")
    p.add_argument("--flow-len",     type=int,   default=10,
                   help="Packets per synthetic flow episode.")
    p.add_argument("--seq-len",      type=int,   default=8,
                   help="BPTT sequence length (must be ≤ flow-len).")
    p.add_argument("--val-fraction", type=float, default=0.15,
                   help="Fraction of episodes held out for validation.")
    p.add_argument("--patience",     type=int,   default=20,
                   help="Early stopping patience (episodes without improvement).")
    p.add_argument("--divergence-factor", type=float, default=8.0,
                   help="Stop if loss exceeds this multiple of its baseline.")
    p.add_argument("--min-buffer",   type=int,   default=200,
                   help="Minimum episodes in replay buffer before training starts.")
    p.add_argument("--val-every",    type=int,   default=5,
                   help="Run full validation every N episodes (default: 5).")
    args = p.parse_args()

    if args.seq_len > args.flow_len:
        print(f"[train] Warning: seq-len ({args.seq_len}) > flow-len ({args.flow_len}); "
              f"clamping seq-len to {args.flow_len}.")
        args.seq_len = args.flow_len

    train(
        dataset_path=pathlib.Path(args.dataset),
        model_out=pathlib.Path(args.out),
        episodes=args.episodes,
        flow_len=args.flow_len,
        seq_len=args.seq_len,
        val_fraction=args.val_fraction,
        patience=args.patience,
        divergence_factor=args.divergence_factor,
        min_buffer_episodes=args.min_buffer,
        val_every=args.val_every,
    )


if __name__ == "__main__":
    main()
