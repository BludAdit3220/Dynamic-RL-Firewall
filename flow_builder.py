"""
flow_builder.py — Synthetic flow-episode generator for the RL firewall.

WHY THIS FILE EXISTS
--------------------
The CIC-IDS2017 dataset ships as *flow-level* CSV rows: each row is an
already-aggregated summary of one TCP/UDP connection (total bytes, IAT
statistics, flag counts, etc.).  This means we cannot reconstruct individual
packet sequences from these files.

To make the environment genuinely sequential (so that gamma=0.99 is justified
and the LSTM agent has real temporal structure to exploit), we synthesise
multi-step episodes from the flow rows:

  * All rows of the **same attack type** (or "BENIGN") are shuffled and then
    chunked into groups of ``flow_len`` consecutive rows.
  * Each group is one synthetic episode: step 0 … step T-1, where T=flow_len.
  * Within the episode, row order is further locally-shuffled (±shuffle_window)
    to prevent the agent from memorising a fixed row ordering.
  * The label for every step in a malicious episode is 1 (the agent must block
    the flow as early as possible).  Every step in a benign episode is 0.

The resulting episodes have genuine temporal correlation:
  - Consecutive steps come from connections of the *same attack family*, so
    feature distributions are locally consistent — just like real packet
    sequences within a flow.
  - s_{t+1} is NOT independent of s_t (same attack cluster), so the LSTM
    hidden state accumulates meaningful context and bootstrapping with
    gamma=0.99 is statistically sound.

EPISODE STRUCTURE
-----------------
Each episode is a dict:
    {
        "features": np.ndarray  shape [T, num_features]  float32
        "labels":   np.ndarray  shape [T]                int32
        "flow_id":  str         human-readable identifier
        "attack_type": str      "BENIGN" or attack name
    }

USAGE
-----
    from flow_builder import build_flow_episodes, split_episodes

    episodes = build_flow_episodes(
        features,          # np.ndarray [N, F]
        labels,            # np.ndarray [N]   (0=benign, 1=attack)
        raw_label_strings, # list[str] of original label text (for grouping)
        flow_len=10,
        seed=42,
    )
    train_eps, val_eps = split_episodes(episodes, val_fraction=0.15, seed=42)
"""
from __future__ import annotations

import random
from typing import List, Dict, Tuple

import numpy as np


# ── Public types ──────────────────────────────────────────────────────────────

Episode = Dict  # keys: features, labels, flow_id, attack_type


# ── Core builder ─────────────────────────────────────────────────────────────

def build_flow_episodes(
    features:          np.ndarray,
    labels:            np.ndarray,
    raw_label_strings: List[str] | None = None,
    *,
    flow_len:          int  = 10,
    shuffle_window:    int  = 3,
    min_flow_len:      int  = 3,
    seed:              int  = 42,
) -> List[Episode]:
    """
    Group rows into synthetic flow episodes.

    Parameters
    ----------
    features : np.ndarray [N, F]
        Pre-processed, *normalised* feature matrix (normalised by the
        caller after fitting on train data only).
    labels : np.ndarray [N]
        Binary labels: 0 = benign, 1 = attack.
    raw_label_strings : list[str] | None
        Original text labels from the CSV (e.g. "BENIGN", "DoS Slowloris",
        "PortScan").  Used to group rows by attack *family* so that
        consecutive steps within an episode share the same attack type.
        If None, groups by binary label only (benign vs attack).
    flow_len : int
        Target number of steps per episode.  Short episodes are padded with
        zeros; very short groups that can't fill even min_flow_len steps are
        discarded.
    shuffle_window : int
        After chunking into episodes, apply a local ±shuffle_window
        perturbation to the row order within each episode (mimics packet
        reordering within a real flow).  Set to 0 to disable.
    min_flow_len : int
        Minimum number of rows a group must have to contribute any episodes.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    List[Episode]
        Shuffled list of episode dicts.
    """
    rng = np.random.default_rng(seed)

    features = np.asarray(features, dtype=np.float32)
    labels   = np.asarray(labels,   dtype=np.int32)
    N, F     = features.shape

    # ── Group rows by attack type (or binary label) ───────────────────────
    if raw_label_strings is not None and len(raw_label_strings) == N:
        group_keys = [str(s).strip() for s in raw_label_strings]
    else:
        group_keys = ["BENIGN" if l == 0 else "ATTACK" for l in labels]

    groups: Dict[str, List[int]] = {}
    for i, key in enumerate(group_keys):
        groups.setdefault(key, []).append(i)

    # ── Build episodes from each group ─────────────────────────────────────
    episodes: List[Episode] = []
    ep_counter = 0

    for attack_type, indices in groups.items():
        if len(indices) < min_flow_len:
            continue  # too few rows to form even a single short episode

        # Shuffle within group so we don't see the same CSV ordering
        idx_arr = np.array(indices, dtype=np.int64)
        rng.shuffle(idx_arr)

        # Chunk into episodes of size flow_len
        for start in range(0, len(idx_arr), flow_len):
            chunk = idx_arr[start : start + flow_len]
            if len(chunk) < min_flow_len:
                break  # discard remainder if too short

            # Local shuffle within the episode (packet reordering)
            if shuffle_window > 0 and len(chunk) > 1:
                chunk = _local_shuffle(chunk, shuffle_window, rng)

            feat_seq  = features[chunk]                      # [T, F]
            label_seq = labels[chunk]                        # [T]

            # Pad to flow_len with zeros if chunk < flow_len
            T = len(chunk)
            if T < flow_len:
                pad_f = np.zeros((flow_len - T, F), dtype=np.float32)
                pad_l = np.zeros(flow_len - T,      dtype=np.int32)
                feat_seq  = np.vstack([feat_seq,  pad_f])
                label_seq = np.concatenate([label_seq, pad_l])

            episodes.append({
                "features":    feat_seq,
                "labels":      label_seq,
                "flow_id":     f"{attack_type}_{ep_counter:06d}",
                "attack_type": attack_type,
                "seq_len":     T,          # actual (un-padded) length
            })
            ep_counter += 1

    # Shuffle episode order
    random.seed(seed)
    random.shuffle(episodes)

    print(
        f"[flow_builder] Built {len(episodes):,} flow episodes "
        f"(flow_len={flow_len}, groups={len(groups)}, "
        f"attack_types={sorted(groups.keys())})"
    )
    return episodes


def split_episodes(
    episodes:     List[Episode],
    val_fraction: float = 0.15,
    seed:         int   = 42,
) -> Tuple[List[Episode], List[Episode]]:
    """
    Stratified train/validation split over episodes.

    Stratifies by ``attack_type`` so both splits have the same mix of
    attack families — critical for fair validation scoring.
    """
    rng = np.random.default_rng(seed)

    # Group episodes by attack type
    groups: Dict[str, List[Episode]] = {}
    for ep in episodes:
        groups.setdefault(ep["attack_type"], []).append(ep)

    train_eps, val_eps = [], []
    for at, eps_list in groups.items():
        arr = list(eps_list)
        rng.shuffle(arr)
        n_val = max(1, int(len(arr) * val_fraction))
        val_eps.extend(arr[:n_val])
        train_eps.extend(arr[n_val:])

    # Re-shuffle both splits
    random.seed(seed + 1)
    random.shuffle(train_eps)
    random.shuffle(val_eps)

    print(
        f"[flow_builder] Split: {len(train_eps):,} train / "
        f"{len(val_eps):,} val episodes "
        f"(val_fraction={val_fraction:.0%}, stratified by attack type)"
    )
    return train_eps, val_eps


# ── Episode-level statistics ──────────────────────────────────────────────────

def episode_stats(episodes: List[Episode]) -> Dict:
    """
    Return a summary dict with counts of benign vs malicious episodes,
    average flow length, and attack-type distribution.
    """
    total       = len(episodes)
    n_malicious = sum(1 for ep in episodes if np.any(ep["labels"] == 1))
    n_benign    = total - n_malicious
    avg_len     = np.mean([ep["seq_len"] for ep in episodes]) if episodes else 0.0

    type_counts: Dict[str, int] = {}
    for ep in episodes:
        type_counts[ep["attack_type"]] = type_counts.get(ep["attack_type"], 0) + 1

    return {
        "total":       total,
        "benign":      n_benign,
        "malicious":   n_malicious,
        "avg_seq_len": float(avg_len),
        "types":       type_counts,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _local_shuffle(arr: np.ndarray, window: int, rng: np.random.Generator) -> np.ndarray:
    """
    Apply a limited local shuffle: each element can move at most ±window
    positions from its original position, simulating packet reordering
    within a network flow without completely destroying temporal order.
    """
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        j = int(rng.integers(max(0, i - window), min(n - 1, i + window) + 1))
        arr[i], arr[j] = arr[j], arr[i]
    return arr
