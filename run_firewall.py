from __future__ import annotations

import argparse
import csv
import datetime as dt
import pathlib
from typing import Dict, Any

import numpy as np
from scapy.all import sniff  # type: ignore

from agent_dqn import DQNAgent, DQNConfig
from iptables_manager import Rule, apply_rule


def build_feature_vector(packet) -> np.ndarray:
    """
    Extract a simple feature vector from a scapy packet.

    For a real deployment you should design richer features that capture
    behavioral characteristics, not just header fields.
    """
    # Basic defaults
    src_ip = packet[0][1].src if hasattr(packet[0][1], "src") else "0.0.0.0"
    dst_ip = packet[0][1].dst if hasattr(packet[0][1], "dst") else "0.0.0.0"
    length = len(packet)

    # Very simple encoding: length + hash buckets for IPs
    def ip_hash(ip: str) -> float:
        return (hash(ip) % 10_000) / 10_000.0

    features = np.array(
        [
            length,
            ip_hash(src_ip),
            ip_hash(dst_ip),
        ],
        dtype=np.float32,
    )
    return features


def packet_to_rule(packet, action: int) -> Rule | None:
    """
    Map an action and packet to an iptables Rule.

    action: 0=allow, 1=block, 2=rate-limit (treated as block in this minimal example).
    """
    if action == 0:
        return None

    layer = packet[0][1]
    src_ip = getattr(layer, "src", None)
    dst_ip = getattr(layer, "dst", None)

    src_port = getattr(layer, "sport", None)
    dst_port = getattr(layer, "dport", None)

    target = "DROP"

    rule = Rule(
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol="tcp",
        target=target,
    )
    return rule


def log_decision(
    log_path: pathlib.Path,
    packet_info: Dict[str, Any],
    action: int,
    dry_run: bool,
):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "src_ip",
                    "dst_ip",
                    "src_port",
                    "dst_port",
                    "length",
                    "action",
                    "dry_run",
                ]
            )
        writer.writerow(
            [
                dt.datetime.utcnow().isoformat(),
                packet_info.get("src_ip"),
                packet_info.get("dst_ip"),
                packet_info.get("src_port"),
                packet_info.get("dst_port"),
                packet_info.get("length"),
                action,
                int(dry_run),
            ]
        )


def run_live_firewall(
    model_path: pathlib.Path,
    log_path: pathlib.Path,
    interface: str,
    dry_run: bool = True,
):
    # Build a dummy config to load the model; state_dim/num_actions are inferred.
    dummy_config = DQNConfig(state_dim=3, num_actions=3)
    agent = DQNAgent(dummy_config)
    agent.load(str(model_path))

    print(f"[run_firewall] Loaded model from {model_path}")
    print(f"[run_firewall] Sniffing on interface={interface}, dry_run={dry_run}")

    def on_packet(packet):
        try:
            features = build_feature_vector(packet)
            action = agent.select_action(features, training=False)

            layer = packet[0][1]
            packet_info = {
                "src_ip": getattr(layer, "src", None),
                "dst_ip": getattr(layer, "dst", None),
                "src_port": getattr(layer, "sport", None),
                "dst_port": getattr(layer, "dport", None),
                "length": len(packet),
            }

            rule = packet_to_rule(packet, action)
            if rule is not None:
                apply_rule(rule, dry_run=dry_run)

            log_decision(log_path, packet_info, action, dry_run=dry_run)
        except Exception as e:  # noqa: BLE001
            print("[run_firewall] Error processing packet:", e)

    sniff(iface=interface, prn=on_packet, store=False)


def main():
    parser = argparse.ArgumentParser(description="Run live RL firewall.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained Keras model.",
    )
    parser.add_argument(
        "--interface",
        type=str,
        default="eth0",
        help="Network interface to sniff on.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="logs/firewall_events.csv",
        help="Path to CSV log file for decisions.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply iptables rules (by default, dry-run only).",
    )
    args = parser.parse_args()

    run_live_firewall(
        model_path=pathlib.Path(args.model),
        log_path=pathlib.Path(args.log),
        interface=args.interface,
        dry_run=not args.apply,
    )


if __name__ == "__main__":
    main()

