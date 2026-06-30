"""
evaluate_vs_ruleset.py — Compare the trained LSTM-DQN firewall against a static ruleset.

FLOW-LEVEL EVALUATION
----------------------
The key improvement over the per-packet evaluation is the addition of
*flow-level* metrics.  A flow is "detected" if the agent blocks it on ANY
step before the terminal packet — just like the live firewall would.

Metrics reported:
  Packet-level:  TP, TN, FP, FN, Precision, Recall, F1, FPR, MCC
  Flow-level:    Flow-DR, Flow-FPR, Early-Detection-Rate
                 (% of attack flows detected in the FIRST HALF of the episode)

The Early-Detection-Rate is the key metric that a stateless classifier cannot
optimise for, because it requires reasoning about the temporal position of a
decision within a flow.  A high Early-DR means the agent is learning to block
attacks before they have fully executed, rather than waiting for the terminal
evidence.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch

from agent_dqn    import LSTMDQNAgent, LSTMDQNConfig
from flow_builder import build_flow_episodes, episode_stats


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_dataset(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, list]:
    if path.is_dir():
        csv_paths = sorted(path.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No CSVs in: {path}")
        df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip() for c in df.columns]
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise KeyError(f"No label column. Columns: {list(df.columns)}")

    raw_lbls = df[label_col].astype(str).str.strip().tolist()
    try:
        labels = df[label_col].to_numpy(dtype=np.int32)
    except (ValueError, TypeError):
        labels = (df[label_col].astype(str).str.strip() != "BENIGN").astype(np.int32).to_numpy()

    features = df.drop(columns=[label_col])
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.dropna(axis=1, how="all")
    features = features.loc[:, features.nunique() > 1]
    feature_names = list(features.columns)
    medians  = features.median()
    features = features.fillna(medians)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(medians)

    return features.to_numpy(dtype=np.float32), labels, raw_lbls, feature_names


# ── Metrics helpers ────────────────────────────────────────────────────────────

def compute_metrics(pred_block: np.ndarray, labels_bin: np.ndarray) -> Dict:
    tp = int(np.sum((pred_block == 1) & (labels_bin == 1)))
    tn = int(np.sum((pred_block == 0) & (labels_bin == 0)))
    fp = int(np.sum((pred_block == 1) & (labels_bin == 0)))
    fn = int(np.sum((pred_block == 0) & (labels_bin == 1)))

    eps       = 1e-9
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    fpr       = fp / (fp + tn + eps)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)
    denom     = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    mcc       = (tp*tn - fp*fn) / (denom + eps)

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy":    round(accuracy,  4),
        "Precision":   round(precision, 4),
        "Recall (DR)": round(recall,    4),
        "F1-Score":    round(f1,        4),
        "FPR":         round(fpr,       4),
        "MCC":         round(mcc,       4),
    }


def print_metrics(name: str, metrics: Dict):
    print(f"\n{'─' * 56}")
    print(f"  {name}")
    print(f"{'─' * 56}")
    print(
        f"  Confusion Matrix:  TP={metrics['TP']:>7,}  FP={metrics['FP']:>7,}\n"
        f"                     FN={metrics['FN']:>7,}  TN={metrics['TN']:>7,}"
    )
    for k, v in metrics.items():
        if k in ("TP", "TN", "FP", "FN"):
            continue
        print(f"  {k:<22} {v}")


# ── LSTM agent evaluation over flows ──────────────────────────────────────────

def evaluate_rl_flows(
    agent:    LSTMDQNAgent,
    episodes: list,
) -> Tuple[Dict, Dict]:
    """
    Evaluate LSTM agent on a list of flow episodes.

    Returns
    -------
    packet_metrics : dict  — packet-level TP/FP/FN/TN + derived metrics
    flow_metrics   : dict  — flow-level TP/FP/FN/TN + early detection rate
    """
    agent.online_network.eval()

    # Packet-level
    all_preds   = []
    all_labels  = []

    # Flow-level
    flow_tp = flow_tn = flow_fp = flow_fn = 0
    early_tp = total_attack_flows = 0

    with torch.no_grad():
        for ep in episodes:
            feats  = ep["features"]
            labels = ep["labels"]
            T      = ep["seq_len"]
            is_malicious = bool(np.any(labels[:T] == 1))

            if is_malicious:
                total_attack_flows += 1

            h, c    = agent.init_hidden()
            blocked = False
            early_blocked = False

            for t in range(T):
                action, h, c = agent.select_action(feats[t], h, c, training=False)
                all_preds.append(1 if action in (1, 2) else 0)
                all_labels.append(int(labels[t]))

                if action in (1, 2):
                    blocked = True
                    if t < T / 2:
                        early_blocked = True
                    # Continue stepping (don't break) so packet-level metrics
                    # are computed for all packets, but mark the flow blocked.

            # Flow-level outcome (was the entire flow correctly classified?)
            if is_malicious:
                if blocked:
                    flow_tp += 1
                    if early_blocked:
                        early_tp += 1
                else:
                    flow_fn += 1
            else:
                if blocked:
                    flow_fp += 1
                else:
                    flow_tn += 1

    agent.online_network.train()

    pkt_metrics = compute_metrics(
        np.array(all_preds,  dtype=np.int32),
        np.array(all_labels, dtype=np.int32),
    )

    eps       = 1e-9
    flow_dr   = flow_tp / max(flow_tp + flow_fn, 1)
    flow_fpr  = flow_fp / max(flow_fp + flow_tn, 1)
    early_dr  = early_tp / max(total_attack_flows, 1)
    flow_prec = flow_tp / max(flow_tp + flow_fp, 1)
    flow_f1   = 2 * flow_dr * flow_prec / (flow_dr + flow_prec + eps)

    flow_metrics = {
        "TP": flow_tp, "TN": flow_tn, "FP": flow_fp, "FN": flow_fn,
        "Flow DR (Recall)":        round(flow_dr,   4),
        "Flow Precision":          round(flow_prec, 4),
        "Flow F1":                 round(flow_f1,   4),
        "Flow FPR":                round(flow_fpr,  4),
        "Early Detection Rate":    round(early_dr,  4),
        "Youden J (Flow)":         round(flow_dr - flow_fpr, 4),
    }

    return pkt_metrics, flow_metrics


# ── Static baseline ruleset ────────────────────────────────────────────────────

def baseline_rule_engine(sample: np.ndarray) -> int:
    """Heuristic: block if pkt_len > 1500 or src-port hash > 0.95."""
    pkt_len  = sample[0]
    port_sig = sample[1] if len(sample) > 1 else 0.0
    if pkt_len > 1500 or port_sig > 0.95:
        return 1
    return 0


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate(model_path: pathlib.Path, dataset_path: pathlib.Path, flow_len: int = 10):
    features, labels, raw_lbls, eval_feat_names = load_dataset(dataset_path)
    n_total   = len(labels)
    n_attacks = int(np.sum(labels == 1))
    n_benign  = n_total - n_attacks
    print(
        f"[evaluate] {n_total:,} rows  "
        f"({n_benign:,} benign, {n_attacks:,} attacks, "
        f"ratio={n_attacks/n_total:.1%})"
    )

    # ── Load normalisation params + feature names (source of truth) ─────────
    norm_path = model_path.parent / "norm_params.json"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"norm_params.json not found at {norm_path}.  "
            "Re-train with run_training.py to generate it."
        )

    with open(norm_path) as f:
        norm = json.load(f)
    mean          = np.array(norm["mean"], dtype=np.float32)
    std           = np.array(norm["std"],  dtype=np.float32)
    num_features  = int(norm["num_features"])
    feature_names = norm.get("feature_names", [])
    if "flow_len" in norm:
        flow_len = norm["flow_len"]

    # ── Align eval features to the training feature set ─────────────────────
    # The training set may have different constant columns than this single CSV.
    # We must select exactly the columns the model was trained on.
    if feature_names:
        feat_name_to_idx = {name: i for i, name in enumerate(eval_feat_names)}
        aligned = np.zeros((len(features), num_features), dtype=np.float32)
        matched = 0
        for col_idx, fname in enumerate(feature_names):
            src_idx = feat_name_to_idx.get(fname, None)
            if src_idx is not None:
                aligned[:, col_idx] = features[:, src_idx]
                matched += 1
            # else: column was constant in training data → stays 0 (safe)
        print(
            f"[evaluate] Feature alignment: {matched}/{num_features} columns matched "
            f"from eval CSV  ({num_features - matched} were constant in training set)"
        )
        features = aligned
    elif features.shape[1] != num_features:
        raise ValueError(
            f"Feature dimension mismatch ({features.shape[1]} vs {num_features}) "
            "and no feature_names saved. Re-train to fix."
        )

    # ── Normalise ─────────────────────────────────────────────────────────
    features = (features - mean) / std
    print("[evaluate] Normalisation applied.")

    # ── Build flow episodes ────────────────────────────────────────────────
    episodes = build_flow_episodes(
        features, labels, raw_lbls, flow_len=flow_len, seed=0
    )
    stats = episode_stats(episodes)
    print(
        f"[evaluate] {stats['total']:,} episodes  "
        f"({stats['benign']:,} benign, {stats['malicious']:,} attack)"
    )

    # ── Load LSTM agent ────────────────────────────────────────────────────
    config = LSTMDQNConfig(state_dim=features.shape[1], num_actions=3)
    agent  = LSTMDQNAgent(config)
    agent.load(str(model_path))
    print(f"[evaluate] Loaded LSTM model from {model_path}")

    # ── RL evaluation (packet + flow level) ───────────────────────────────
    rl_pkt_metrics, rl_flow_metrics = evaluate_rl_flows(agent, episodes)

    # ── Static baseline (packet-level only) ───────────────────────────────
    base_block = np.array(
        [baseline_rule_engine(x) for x in features], dtype=np.int32
    )
    base_metrics = compute_metrics(base_block, labels)

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "═" * 56)
    print("  LSTM-DQN RL FIREWALL vs STATIC RULESET")
    print("═" * 56)

    print_metrics("RL Agent — Packet-Level",   rl_pkt_metrics)
    print_metrics("Static Baseline Ruleset",   base_metrics)

    print(f"\n{'─' * 56}")
    print("  RL Agent — Flow-Level (LSTM temporal decisions)")
    print(f"{'─' * 56}")
    print(
        f"  Confusion Matrix:  TP={rl_flow_metrics['TP']:>6,}  "
        f"FP={rl_flow_metrics['FP']:>6,}\n"
        f"                     FN={rl_flow_metrics['FN']:>6,}  "
        f"TN={rl_flow_metrics['TN']:>6,}"
    )
    for k, v in rl_flow_metrics.items():
        if k in ("TP", "TN", "FP", "FN"):
            continue
        print(f"  {k:<28} {v}")

    print(
        f"\n  ► Early-Detection-Rate is the key metric that a stateless\n"
        f"    classifier cannot optimise: it measures the fraction of\n"
        f"    attack flows blocked BEFORE the terminal packet arrives.\n"
        f"    A high value means the agent learned to block early using\n"
        f"    LSTM context — something impossible without temporal memory."
    )


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate LSTM RL firewall vs static ruleset.")
    p.add_argument("--model",    type=str, required=True)
    p.add_argument("--dataset",  type=str, required=True)
    p.add_argument("--flow-len", type=int, default=10,
                   help="Episode length used when building flow episodes for eval.")
    args = p.parse_args()

    evaluate(
        model_path=pathlib.Path(args.model),
        dataset_path=pathlib.Path(args.dataset),
        flow_len=args.flow_len,
    )


if __name__ == "__main__":
    main()
