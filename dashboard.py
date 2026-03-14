import pathlib

import pandas as pd
import streamlit as st


def load_logs(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
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
    return pd.read_csv(path)


def main():
    st.set_page_config(
        page_title="Dynamic RL Firewall Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Dynamic Reinforcement Learning Firewall")
    st.caption("Real-time traffic decisions and policy adaptation")

    log_path_str = st.sidebar.text_input(
        "Log file path", value="logs/firewall_events.csv"
    )
    log_path = pathlib.Path(log_path_str)

    refresh_sec = st.sidebar.slider("Auto-refresh (seconds)", 5, 60, 10)
    st_autorefresh = st.sidebar.checkbox("Auto-refresh", value=True)

    if st_autorefresh:
        st.experimental_rerun  # type: ignore[attr-defined]

    df = load_logs(log_path)

    if df.empty:
        st.info("No firewall events logged yet.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=False)

    st.subheader("Recent Decisions")
    st.dataframe(df.head(200), use_container_width=True)

    st.subheader("Traffic Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Events", len(df))

    with col2:
        blocked = (df["action"] == 1).sum()
        st.metric("Blocked / Rate-limited", int(blocked))

    with col3:
        allow = (df["action"] == 0).sum()
        st.metric("Allowed", int(allow))

    st.subheader("Actions Over Time")
    chart_df = (
        df.set_index("timestamp")
        .groupby([pd.Grouper(freq="1Min"), "action"])
        .size()
        .reset_index(name="count")
    )
    chart_pivot = chart_df.pivot_table(
        index="timestamp", columns="action", values="count", fill_value=0
    )
    chart_pivot.columns = [f"action_{c}" for c in chart_pivot.columns]
    st.line_chart(chart_pivot)

    st.subheader("Top Source IPs")
    top_src = (
        df.groupby("src_ip")
        .size()
        .sort_values(ascending=False)
        .head(20)
        .reset_index(name="events")
    )
    st.bar_chart(top_src.set_index("src_ip"))


if __name__ == "__main__":
    main()

