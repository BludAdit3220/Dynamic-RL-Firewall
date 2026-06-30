"""
run_firewall.py — Stateful LSTM-DQN firewall with real CIC-IDS feature parity.

WHAT CHANGED FROM THE PREVIOUS VERSION
----------------------------------------
The previous version fed the model only 8 crude packet-header features
(length, IP hash, port, protocol flags) zero-padded to num_features.  Because
the model was trained on 78 CIC-IDS flow-statistics features (IAT mean/std,
flag counts, byte rates, window sizes, …), those Q-values were meaningless.

This version introduces FlowStatAccumulator (flow_stats.py):
  - One accumulator per tracked 5-tuple flow
  - Updated on every incoming packet with Welford's online stats
  - Produces the same 78-feature CIC-IDS vector the model was trained on
  - feature_names from norm_params.json selects exactly the columns that
    survived the training-time cleaning step (constant/NaN column drops)

The model now receives feature vectors that are in-distribution relative to
its training data, so the Q-values are genuinely informative.

PIPELINE (per packet)
---------------------
  scapy packet
    → extract 5-tuple → look up / create FlowEntry
    → FlowEntry.accumulator.update(packet)          (update running stats)
    → accumulator.to_feature_vector(feature_names)  (build 78-col CIC vector)
    → normalise with saved mean/std
    → agent.select_action(features, h, c)           (LSTM forward step)
    → update h, c in FlowEntry
    → if action in {block, rate-limit}: apply iptables rule

FLOW IDLE EVICTION
------------------
Flows that have received no packet for FLOW_IDLE_TIMEOUT seconds are evicted
from the state table.  This bounds memory use on long-running deployments.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scapy.all import sniff, IP, IPv6, TCP, UDP   # type: ignore
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[run_firewall] WARNING: scapy not installed. Live sniffing unavailable.")

from agent_dqn     import LSTMDQNAgent, LSTMDQNConfig
from flow_stats    import FlowStatAccumulator
from iptables_manager import Rule, apply_rule


# ── Constants ─────────────────────────────────────────────────────────────────

FLOW_IDLE_TIMEOUT = 60.0   # seconds of silence before a flow state is evicted


# ── Per-flow state entry ──────────────────────────────────────────────────────

@dataclass
class FlowEntry:
    accumulator: FlowStatAccumulator
    h:           "torch.Tensor"      # LSTM hidden state
    c:           "torch.Tensor"      # LSTM cell state
    step:        int = 0
    last_seen:   float = field(default_factory=time.monotonic)
    blocked:     bool = False        # True once we've issued a block rule


# ── Packet helpers ─────────────────────────────────────────────────────────────

def extract_packet_meta(packet) -> Dict[str, Any]:
    """Extract human-readable 5-tuple metadata from a scapy packet."""
    from scapy.all import IP as _IP, IPv6 as _IPv6, TCP as _TCP, UDP as _UDP

    ip_layer = None
    if packet.haslayer(_IP):
        ip_layer = packet[_IP]
    elif packet.haslayer(_IPv6):
        ip_layer = packet[_IPv6]

    src_ip   = getattr(ip_layer, "src", None)
    dst_ip   = getattr(ip_layer, "dst", None)
    src_port = dst_port = None
    proto    = "other"

    if packet.haslayer(_TCP):
        src_port = packet[_TCP].sport
        dst_port = packet[_TCP].dport
        proto    = "tcp"
    elif packet.haslayer(_UDP):
        src_port = packet[_UDP].sport
        dst_port = packet[_UDP].dport
        proto    = "udp"

    return {
        "src_ip":   src_ip,
        "dst_ip":   dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "proto":    proto,
        "length":   len(packet),
    }


def flow_key(meta: Dict[str, Any]) -> Tuple:
    """Canonical 5-tuple — forward direction is defined by the first packet."""
    return (meta["src_ip"], meta["dst_ip"],
            meta["src_port"], meta["dst_port"], meta["proto"])


def make_rule(meta: Dict[str, Any], action: int) -> Optional[Rule]:
    if action == 0:
        return None
    return Rule(
        src_ip=meta["src_ip"],
        dst_ip=meta["dst_ip"],
        src_port=meta["src_port"],
        dst_port=meta["dst_port"],
        protocol=meta.get("proto", "tcp"),
        target="DROP",
    )


# ── Decision logger ───────────────────────────────────────────────────────────

def log_decision(
    log_path:  pathlib.Path,
    meta:      Dict[str, Any],
    action:    int,
    flow_id:   str,
    flow_step: int,
    dry_run:   bool,
):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    with log_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow([
                "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
                "proto", "length", "action", "flow_id", "flow_step", "dry_run",
            ])
        w.writerow([
            dt.datetime.utcnow().isoformat(),
            meta["src_ip"], meta["dst_ip"],
            meta["src_port"], meta["dst_port"],
            meta["proto"], meta["length"],
            action, flow_id, flow_step, int(dry_run),
        ])


# ── FlowTracker (stateful LSTM + CIC-IDS feature accumulation) ────────────────

class FlowTracker:
    """
    Maintains per-flow LSTM hidden state AND incremental CIC-IDS feature
    statistics.  This is the core of the stateful live firewall.

    For each 5-tuple (src_ip, dst_ip, src_port, dst_port, proto):
      - A FlowStatAccumulator computes the CIC-IDS features incrementally
      - An LSTM hidden state (h, c) carries temporal context across packets
      - A step counter tracks how many packets this flow has been seen

    On each packet:
      1. Accumulator updated → CIC-IDS feature vector produced
      2. Normalised with training mean/std
      3. Passed to LSTM agent → action + new (h, c)
      4. (h, c) stored back for the next packet of the same flow
    """

    def __init__(
        self,
        agent:         LSTMDQNAgent,
        mean:          np.ndarray,
        std:           np.ndarray,
        feature_names: List[str],
        idle_timeout:  float = FLOW_IDLE_TIMEOUT,
    ):
        self.agent         = agent
        self.mean          = mean
        self.std           = std
        self.feature_names = feature_names
        self.idle_timeout  = idle_timeout
        self._table: Dict[Tuple, FlowEntry] = {}

    def process(self, key: Tuple, packet, dst_port: int) -> Tuple[int, str, int]:
        """
        Process one packet for the given flow key.

        Returns (action, flow_id, flow_step).
        """
        now = time.monotonic()
        self._evict_idle(now)

        if key not in self._table:
            h, c = self.agent.init_hidden()
            self._table[key] = FlowEntry(
                accumulator=FlowStatAccumulator(dst_port=dst_port),
                h=h,
                c=c,
                last_seen=now,
            )

        entry = self._table[key]
        entry.last_seen = now

        # If already blocked, maintain state but skip re-issuing the rule
        if entry.blocked:
            entry.accumulator.update(packet)  # keep stats current
            return 1, self._flow_id(key, entry.step), entry.step

        # ── Update accumulator with this packet ──────────────────────────
        entry.accumulator.update(packet)

        # ── Build feature vector (CIC-IDS features, selected & normalised) ─
        raw = entry.accumulator.to_feature_vector(self.feature_names)
        obs = (raw - self.mean) / self.std

        # ── LSTM step ────────────────────────────────────────────────────
        action, h_new, c_new = self.agent.select_action(
            obs, entry.h, entry.c, training=False
        )
        entry.h    = h_new
        entry.c    = c_new
        entry.step += 1

        if action in (1, 2):
            entry.blocked = True

        return action, self._flow_id(key, entry.step), entry.step

    def _evict_idle(self, now: float):
        stale = [k for k, e in self._table.items()
                 if now - e.last_seen > self.idle_timeout]
        for k in stale:
            del self._table[k]

    @staticmethod
    def _flow_id(key: Tuple, step: int) -> str:
        src_ip, dst_ip, src_port, dst_port, proto = key
        return f"{src_ip}:{src_port}\u2192{dst_ip}:{dst_port}/{proto}@{step}"

    def active_flows(self) -> int:
        return len(self._table)


# ── Live firewall loop ─────────────────────────────────────────────────────────

def run_live_firewall(
    model_path: pathlib.Path,
    log_path:   pathlib.Path,
    interface:  str,
    dry_run:    bool = True,
):
    if not SCAPY_AVAILABLE:
        raise RuntimeError("scapy required. Install with: pip install scapy")

    # ── Load normalisation params + feature names ──────────────────────────
    norm_path = model_path.parent / "norm_params.json"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"norm_params.json not found at {norm_path}. "
            "Run run_training.py first to generate it."
        )

    with open(norm_path) as f:
        norm = json.load(f)

    mean          = np.array(norm["mean"], dtype=np.float32)
    std           = np.array(norm["std"],  dtype=np.float32)
    num_features  = int(norm["num_features"])
    feature_names = norm.get("feature_names", [])

    if not feature_names:
        print(
            "[run_firewall] WARNING: feature_names missing from norm_params.json. "
            "Re-train with the updated run_training.py to get proper feature alignment. "
            "Falling back to raw 8-feature extraction."
        )

    print(
        f"[run_firewall] Loaded norm params: "
        f"num_features={num_features}, "
        f"feature_names={'saved (' + str(len(feature_names)) + ' cols)' if feature_names else 'MISSING — retrain!'}"
    )

    # ── Load LSTM agent ────────────────────────────────────────────────────
    config = LSTMDQNConfig(state_dim=num_features, num_actions=3)
    agent  = LSTMDQNAgent(config)
    agent.load(str(model_path))
    agent.online_network.eval()
    print(f"[run_firewall] Loaded LSTM model from {model_path}")
    print(f"[run_firewall] Interface={interface!r}  dry_run={dry_run}")
    print(f"[run_firewall] Flow idle timeout: {FLOW_IDLE_TIMEOUT}s")

    tracker = FlowTracker(
        agent=agent,
        mean=mean,
        std=std,
        feature_names=feature_names,
        idle_timeout=FLOW_IDLE_TIMEOUT,
    )

    def on_packet(packet):
        try:
            meta  = extract_packet_meta(packet)
            fkey  = flow_key(meta)
            dport = meta["dst_port"] or 0

            action, fid, fstep = tracker.process(fkey, packet, dst_port=dport)

            if action in (1, 2):
                rule = make_rule(meta, action)
                if rule is not None:
                    apply_rule(rule, dry_run=dry_run)
                src = f"{meta['src_ip']}:{meta['src_port']}"
                print(
                    f"[run_firewall] {'[DRY] ' if dry_run else ''}"
                    f"{'BLOCK' if action == 1 else 'RATE-LIMIT'} {src} "
                    f"(flow step {fstep}, active flows: {tracker.active_flows()})"
                )

            log_decision(log_path, meta, action, fid, fstep, dry_run=dry_run)

        except Exception as e:
            print(f"[run_firewall] Error processing packet: {e}")

    print("[run_firewall] Sniffing — press Ctrl+C to stop.")
    sniff(iface=interface, prn=on_packet, store=False)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run stateful LSTM-DQN RL firewall with CIC-IDS feature parity."
    )
    parser.add_argument("--model",     type=str, required=True,
                        help="Path to trained .pt model file.")
    parser.add_argument("--interface", type=str, default="eth0",
                        help="Network interface to sniff on.")
    parser.add_argument("--log",       type=str, default="logs/firewall_events.csv",
                        help="CSV log output path.")
    parser.add_argument("--apply",     action="store_true",
                        help="Apply iptables rules (default: dry-run).")
    args = parser.parse_args()

    run_live_firewall(
        model_path=pathlib.Path(args.model),
        log_path=pathlib.Path(args.log),
        interface=args.interface,
        dry_run=not args.apply,
    )


if __name__ == "__main__":
    main()
