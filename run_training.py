from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import numpy as np

from firewall_env import DynamicFirewallEnv
from agent_dqn import DQNAgent, DQNConfig


def load_dataset(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a preprocessed dataset of traffic features and labels.

    Supports either:
    - A single CSV file
    - A directory containing multiple CSV files (all ``*.csv`` will be concatenated)

    Expected CSV format:
        f1,f2,...,fN,label

    Where label is 0 for benign, 1 for malicious.
    """
    import pandas as pd

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

    # CIC-IDS2017 uses "Label" with string values like "BENIGN" / attack name.
    label_col = None
    if "label" in df.columns:
        label_col = "label"
    elif "Label" in df.columns:
        label_col = "Label"
    else:
        raise KeyError(f"No 'label' / 'Label' column found in dataset. Columns: {list(df.columns)}")

    # Binary label: 0 = benign, 1 = attack (anything not exactly "BENIGN")
    labels_raw = df[label_col]
    if labels_raw.dtype == object:
        labels = (labels_raw != "BENIGN").astype(np.int32).to_numpy()
    else:
        labels = labels_raw.to_numpy(dtype=np.int32)

    features = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    return features, labels


def train(
    dataset_path: pathlib.Path,
    model_out: pathlib.Path,
    episodes: int = 50,
    max_steps_per_episode: int = 10_000,
):
    features, labels = load_dataset(dataset_path)

    env = DynamicFirewallEnv(traffic_samples=features, labels=labels)

    config = DQNConfig(
        state_dim=features.shape[1],
        num_actions=env.action_space.n,
    )
    agent = DQNAgent(config)

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()

            state = next_state
            episode_reward += reward

            if done:
                break

        print(
            f"Episode {episode + 1}/{episodes} "
            f"- reward={episode_reward:.2f} "
            f"- epsilon={agent.epsilon:.3f}"
        )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(model_out))
    print(f"Saved trained model to {model_out}")


def main():
    parser = argparse.ArgumentParser(description="Train RL-based firewall (DQN).")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to CSV dataset with traffic features and labels.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models/rl_firewall_dqn",
        help="Output path for trained Keras model.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of training episodes.",
    )
    args = parser.parse_args()

    train(
        dataset_path=pathlib.Path(args.dataset),
        model_out=pathlib.Path(args.out),
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()

