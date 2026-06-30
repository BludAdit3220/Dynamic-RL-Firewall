"""
dashboard.py — Streamlit dashboard for the LSTM-DQN RL firewall.

NEW in the flow-level version:
  1. "Active Flow Intelligence" panel: shows tracked flows grouped by
     flow_id with step counts and action distributions.
  2. "Early Detection Timeline": bar chart showing what fraction of blocked
     flows were detected at each packet step — the key metric that shows
     the LSTM is doing temporal reasoning, not just per-packet classification.
  3. Extended log format: new columns flow_id, flow_step, proto are shown.
  4. All existing panels (summary metrics, actions over time, top source IPs)
     are retained and improved.
"""
from __future__ import annotations

import pathlib
import time
from typing import Optional

import pandas as pd
import streamlit as st


# ── Constants ─────────────────────────────────────────────────────────────────

ACTION_LABELS  = {0: "Allow", 1: "Block", 2: "Rate-limit"}
ACTION_COLOURS = {0: "#2ecc71", 1: "#e74c3c", 2: "#f39c12"}

EXPECTED_COLUMNS = [
    "timestamp", "src_ip", "dst_ip",
    "src_port", "dst_port", "proto", "length",
    "action", "flow_id", "flow_step", "dry_run",
]

# Legacy log columns (old format had no flow_id / flow_step)
LEGACY_COLUMNS = [
    "timestamp", "src_ip", "dst_ip",
    "src_port", "dst_port", "length", "action", "dry_run",
]


# ── Log loading ────────────────────────────────────────────────────────────────

def load_logs(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not read log file: {e}")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    # Ensure all columns exist (back-compat with old log format)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df


# ── Main dashboard ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Dynamic RL Firewall Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🛡️ LSTM-DQN Dynamic Reinforcement Learning Firewall")
    st.caption(
        "Flow-level stateful traffic decisions — the agent carries LSTM memory "
        "across packets within each connection."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        log_path_str = st.text_input("Log file path", value="logs/firewall_events.csv")
        log_path     = pathlib.Path(log_path_str)

        refresh_sec  = st.slider("Auto-refresh interval (s)", 5, 60, 10)
        auto_refresh = st.checkbox("Auto-refresh enabled", value=True)

        st.markdown("---")
        st.markdown("**Action legend**")
        for code, label in ACTION_LABELS.items():
            colour = ACTION_COLOURS[code]
            st.markdown(
                f"<span style='color:{colour}'>●</span> {code} — {label}",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            "**About the LSTM agent**\n\n"
            "Each row in the log is one *packet* within a tracked flow. "
            "The `flow_step` column shows which packet in the flow this is. "
            "The agent carries hidden state across packets — so its decision "
            "on packet 5 depends on what it saw on packets 1–4."
        )

    # ── Auto-refresh ──────────────────────────────────────────────────────
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()

    if auto_refresh:
        elapsed = time.time() - st.session_state["last_refresh"]
        if elapsed >= refresh_sec:
            st.session_state["last_refresh"] = time.time()
            st.rerun()
        st.info(f"Auto-refresh in ~{max(0, int(refresh_sec - elapsed))}s — press R to refresh.")

    if st.button("🔄 Refresh now"):
        st.session_state["last_refresh"] = time.time()
        st.rerun()

    # ── Load data ─────────────────────────────────────────────────────────
    df = load_logs(log_path)

    if df.empty:
        st.info(
            "No firewall events logged yet. "
            "Start `run_firewall.py` and events will appear here."
        )
        return

    df["timestamp"]    = pd.to_datetime(df["timestamp"], errors="coerce")
    df                 = df.dropna(subset=["timestamp"])
    df                 = df.sort_values("timestamp", ascending=False)
    df["action_label"] = df["action"].map(ACTION_LABELS).fillna("Unknown")

    # Coerce flow_step to int where available
    if df["flow_step"].notna().any():
        df["flow_step"] = pd.to_numeric(df["flow_step"], errors="coerce")

    # ── Summary metrics ───────────────────────────────────────────────────
    st.subheader("📊 Summary")
    total     = len(df)
    n_allow   = int((df["action"] == 0).sum())
    n_block   = int((df["action"] == 1).sum())
    n_ratelim = int((df["action"] == 2).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events",   f"{total:,}")
    col2.metric("✅ Allowed",      f"{n_allow:,}",   f"{n_allow/total:.1%}")
    col3.metric("🚫 Blocked",      f"{n_block:,}",   f"{n_block/total:.1%}")
    col4.metric("⏬ Rate-limited", f"{n_ratelim:,}", f"{n_ratelim/total:.1%}")

    # Optional detection-rate panel (labelled logs)
    if "label" in df.columns and df["label"].notna().any():
        st.subheader("🎯 Detection Estimates (from labelled log)")
        lbl_df          = df.dropna(subset=["label"])
        lbl_df["label"] = lbl_df["label"].astype(int)
        blocked         = (lbl_df["action"] >= 1).astype(int)
        tp  = int(((blocked == 1) & (lbl_df["label"] == 1)).sum())
        fn  = int(((blocked == 0) & (lbl_df["label"] == 1)).sum())
        fp  = int(((blocked == 1) & (lbl_df["label"] == 0)).sum())
        tn  = int(((blocked == 0) & (lbl_df["label"] == 0)).sum())
        dr  = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        mc1, mc2 = st.columns(2)
        mc1.metric("Detection Rate (Recall)", f"{dr:.1%}")
        mc2.metric("False Positive Rate",     f"{fpr:.1%}")

    # ── Flow Intelligence Panel ────────────────────────────────────────────
    if df["flow_id"].notna().any():
        st.subheader("🔗 Flow-Level Intelligence")
        st.caption(
            "Each row is a unique tracked connection. "
            "`max_step` = how many packets were inspected before a decision. "
            "Low max_step on blocked flows = LSTM caught the attack early."
        )

        flow_df = df[df["flow_id"].notna()].copy()
        flow_summary = (
            flow_df.groupby("flow_id")
            .agg(
                packets    = ("flow_step", "count"),
                max_step   = ("flow_step", "max"),
                final_action = ("action", "last"),
                src_ip     = ("src_ip",   "first"),
                dst_ip     = ("dst_ip",   "first"),
            )
            .reset_index()
        )
        flow_summary["final_action"] = flow_summary["final_action"].map(ACTION_LABELS)
        n_flows   = len(flow_summary)
        n_blocked_flows = int((flow_summary["final_action"] == "Block").sum())
        col_a, col_b = st.columns(2)
        col_a.metric("Unique Flows Tracked", f"{n_flows:,}")
        col_b.metric("Flows Blocked",        f"{n_blocked_flows:,}",
                     f"{n_blocked_flows/max(n_flows,1):.1%}")
        st.dataframe(flow_summary.head(50), use_container_width=True)

        # ── Early-Detection Timeline ───────────────────────────────────────
        if "flow_step" in flow_df.columns:
            st.subheader("⚡ Early Detection Timeline")
            st.caption(
                "When in the flow (which packet number) did the LSTM agent "
                "decide to block?  A spike at low step numbers means the agent "
                "is learning to detect attacks *early* using accumulated context "
                "— something a stateless classifier cannot do."
            )
            blocked_flows = flow_df[flow_df["action"].isin([1, 2])].copy()
            if not blocked_flows.empty:
                first_block = (
                    blocked_flows.groupby("flow_id")["flow_step"]
                    .min()
                    .reset_index(name="detection_step")
                )
                step_counts = (
                    first_block["detection_step"]
                    .value_counts()
                    .sort_index()
                    .reset_index()
                )
                step_counts.columns = ["Packet Step", "Flows Blocked"]
                st.bar_chart(step_counts.set_index("Packet Step"))
            else:
                st.info("No blocked flows yet — agent is still in explore/allow mode.")

    # ── Actions Over Time ──────────────────────────────────────────────────
    st.subheader("📈 Actions Over Time")
    try:
        chart_df = (
            df.set_index("timestamp")
            .groupby([pd.Grouper(freq="1min"), "action"])
            .size()
            .reset_index(name="count")
        )
        if not chart_df.empty:
            pivot = chart_df.pivot_table(
                index="timestamp", columns="action",
                values="count", fill_value=0,
            )
            pivot.columns = [
                ACTION_LABELS.get(int(c), f"action_{c}")
                for c in pivot.columns
            ]
            st.line_chart(pivot)
        else:
            st.info("Not enough data for the time-series chart yet.")
    except Exception as e:
        st.warning(f"Could not render time-series chart: {e}")

    # ── Recent decisions table ─────────────────────────────────────────────
    st.subheader("📋 Recent Decisions (last 200)")
    display_cols = [
        c for c in [
            "timestamp", "src_ip", "dst_ip",
            "src_port", "dst_port", "proto", "length",
            "action_label", "flow_id", "flow_step", "dry_run",
        ]
        if c in df.columns
    ]
    st.dataframe(df[display_cols].head(200), use_container_width=True)

    # ── Top source IPs ─────────────────────────────────────────────────────
    st.subheader("🌐 Top Source IPs (by event count)")
    if df["src_ip"].notna().any():
        top_src = (
            df.groupby("src_ip")
            .size()
            .sort_values(ascending=False)
            .head(20)
            .reset_index(name="events")
        )
        st.bar_chart(top_src.set_index("src_ip"))
    else:
        st.info("No source IP data available yet.")


if __name__ == "__main__":
    main()
