from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd

from agent_dqn import DQNAgent, DQNConfig


def load_dataset(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a preprocessed dataset of traffic features and labels.

    Supports either:
    - A single CSV file
    - A directory containing multiple CSV files (all ``*.csv`` will be concatenated)
    """
    if path.is_dir():
        csv_paths = sorted(path.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        dfs = [pd.read_csv(p) for p in csv_paths]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(path)

    # Normalize column names (strip spaces, unify case)
    df.columns = [c.strip() for c in df.columns]

    label_col = None
    if "label" in df.columns:
        label_col = "label"
    elif "Label" in df.columns:
        label_col = "Label"
    else:
        raise KeyError(f"No 'label' / 'Label' column found in dataset. Columns: {list(df.columns)}")

    labels_raw = df[label_col]
    if labels_raw.dtype == object:
        labels = (labels_raw != "BENIGN").astype(np.int32).to_numpy()
    else:
        labels = labels_raw.to_numpy(dtype=np.int32)

    features = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    return features, labels


def baseline_rule_engine(sample: np.ndarray) -> int:
    """
    Very simple static ruleset:
      - If feature[0] (length) > threshold, treat as suspicious and block.
      - Otherwise allow.

    Returns action: 0=allow, 1=block.
    """
    length = sample[0]
    if length > 1500:  # example MTU-ish threshold
        return 1
    return 0


def evaluate(
    model_path: pathlib.Path,
    dataset_path: pathlib.Path,
):
    features, labels = load_dataset(dataset_path)

    # Load agent
    dummy_config = DQNConfig(state_dim=features.shape[1], num_actions=3)
    agent = DQNAgent(dummy_config)
    agent.load(str(model_path))

    # Metrics
    def compute_metrics(pred_actions: np.ndarray, labels_bin: np.ndarray, positive_action=1):
        # Map RL actions 0/1/2 to binary "block or not"
        if pred_actions.ndim == 1:
            pred_block = (pred_actions == positive_action).astype(int)
        else:
            raise ValueError("pred_actions must be 1D")

        tp = np.sum((pred_block == 1) & (labels_bin == 1))
        tn = np.sum((pred_block == 0) & (labels_bin == 0))
        fp = np.sum((pred_block == 1) & (labels_bin == 0))
        fn = np.sum((pred_block == 0) & (labels_bin == 1))

        detection_rate = tp / (tp + fn + 1e-9)
        false_positive_rate = fp / (fp + tn + 1e-9)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)

        return {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "detection_rate": detection_rate,
            "false_positive_rate": false_positive_rate,
            "accuracy": accuracy,
        }

    # Evaluate RL agent
    rl_actions = []
    for x in features:
        a = agent.select_action(x, training=False)
        # Treat both 1=block and 2=rate-limit as "block" for binary comparison
        rl_actions.append(1 if a in (1, 2) else 0)
    rl_actions = np.array(rl_actions, dtype=np.int32)

    # Evaluate baseline ruleset
    baseline_actions = np.array([baseline_rule_engine(x) for x in features], dtype=np.int32)

    # Binary labels: 1=attack, 0=benign
    labels_bin = labels.copy()

    rl_metrics = compute_metrics(rl_actions, labels_bin, positive_action=1)
    base_metrics = compute_metrics(baseline_actions, labels_bin, positive_action=1)

    print("=== RL Firewall vs Static Ruleset ===")
    print("RL Firewall:")
    for k, v in rl_metrics.items():
        print(f"  {k}: {v}")

    print("\nStatic Ruleset:")
    for k, v in base_metrics.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare RL firewall against baseline static ruleset."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained Keras model.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset CSV (same format as training).",
    )
    args = parser.parse_args()

    evaluate(model_path=pathlib.Path(args.model), dataset_path=pathlib.Path(args.dataset))


if __name__ == "__main__":
    main()

