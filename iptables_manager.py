import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class Rule:
    """
    Simple representation of a dynamic firewall rule.

    This can be extended to include rate-limiting via tc, advanced matches, etc.
    """

    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    protocol: str = "tcp"  # tcp/udp/any
    target: str = "DROP"  # ACCEPT/DROP/REJECT


def _build_match_args(rule: Rule) -> list[str]:
    args: list[str] = []
    if rule.protocol and rule.protocol.lower() != "any":
        args += ["-p", rule.protocol.lower()]
    if rule.src_ip:
        args += ["-s", rule.src_ip]
    if rule.dst_ip:
        args += ["-d", rule.dst_ip]
    if rule.src_port:
        args += ["--sport", str(rule.src_port)]
    if rule.dst_port:
        args += ["--dport", str(rule.dst_port)]
    return args


def apply_rule(rule: Rule, chain: str = "INPUT", dry_run: bool = True):
    """
    Apply an iptables rule corresponding to a policy decision.

    WARNING: Running this with dry_run=False will modify the system firewall.
    Make sure you fully understand the impact and have console access.
    """
    cmd = ["iptables", "-A", chain] + _build_match_args(rule) + ["-j", rule.target]
    if dry_run:
        print("[iptables_manager] Dry run:", " ".join(cmd))
        return

    subprocess.run(cmd, check=True)


def clear_dynamic_rules(chain: str = "INPUT", dry_run: bool = True, comment_tag: str = "RLFW"):
    """
    Clear dynamic rules previously installed by the RL firewall.

    For a production system you would tag rules with comments using iptables -m comment
    and then delete only those rules. Here we expose a coarse helper that can be
    extended as needed.
    """
    # Example implementation using iptables-save/restore could go here.
    # For safety, we print a note instead of wiping rules.
    if dry_run:
        print(
            f"[iptables_manager] Requested clearing dynamic rules on chain={chain}, "
            f"comment_tag={comment_tag} (dry run - no changes made)."
        )
        return

    raise NotImplementedError(
        "Clearing dynamic rules non-destructively is environment-specific. "
        "Implement this carefully for your deployment."
    )

