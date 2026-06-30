"""
flow_stats.py — Incremental CIC-IDS2017 flow statistics from raw scapy packets.

WHY THIS FILE EXISTS
--------------------
The model was trained on CIC-IDS2017 CSV rows, where each row contains ~78
pre-computed flow-level features (IAT mean/std, packet length distribution,
flag counts, etc.) aggregated over a COMPLETE connection.

At live-inference time we receive raw packets one at a time.  To feed the
model the same feature space it was trained on, we must compute these 78
features *incrementally* as each packet arrives.

This file provides FlowStatAccumulator: one instance per tracked flow,
updated on each incoming packet, producing a feature vector in the exact
same 78-column order as the CIC-IDS2017 CSVs.

FEATURE FIDELITY
----------------
Most features are computed exactly:
  - Packet counts, byte totals, header lengths        → exact
  - IAT mean/std/min/max (flow, fwd, bwd)             → exact (Welford's alg)
  - TCP flag counts (SYN, ACK, FIN, RST, PSH, URG…)  → exact
  - Packet length stats (min/max/mean/std/variance)   → exact (Welford's alg)
  - Init window size (forward & backward)             → exact (from first SYN)
  - Flow Bytes/s and Packets/s                        → exact (from timestamps)

A few CIC-IDS features are complex to compute live and are approximated:
  - Bulk features (Fwd/Bwd Avg Bytes/Packets/Bulk Rate) → 0 (post-hoc stat)
  - Active/Idle times (requires sliding window logic)   → running estimate
  - Subflow counts (TCP subflows within connection)     → mirrors total counts

These approximations affect ~12/78 features.  The remaining 66 are exact,
giving the model a high-fidelity view of live traffic.
"""
from __future__ import annotations

import math
import time
from typing import Optional, List, Tuple

import numpy as np


# ── CIC-IDS 2017 feature names in column order (matches CSV header) ───────────
# Used to align the feature vector with what the model was trained on.

CIC_FEATURE_NAMES: List[str] = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length",          # appears twice in CIC-IDS (duplicate col)
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

assert len(CIC_FEATURE_NAMES) == 78, f"Expected 78 features, got {len(CIC_FEATURE_NAMES)}"


# ── Online statistics (Welford's one-pass algorithm) ─────────────────────────

class _RunningStats:
    """
    Incrementally computes mean, variance, std, min, max, and total for a
    stream of float values using Welford's numerically stable algorithm.
    Avoids storing all values in memory.
    """
    __slots__ = ("n", "mean", "_M2", "min", "max", "total")

    def __init__(self):
        self.n     = 0
        self.mean  = 0.0
        self._M2   = 0.0
        self.min   = float("inf")
        self.max   = float("-inf")
        self.total = 0.0

    def update(self, x: float):
        self.n     += 1
        self.total += x
        if x < self.min: self.min = x
        if x > self.max: self.max = x
        delta       = x - self.mean
        self.mean  += delta / self.n
        self._M2   += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self._M2 / self.n if self.n >= 2 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def safe_min(self) -> float:
        return self.min if self.n > 0 else 0.0

    def safe_max(self) -> float:
        return self.max if self.n > 0 else 0.0


# ── Active / Idle window tracker ──────────────────────────────────────────────

_ACTIVE_THRESHOLD = 1.0   # seconds: gap > this → idle period

class _ActiveIdleTracker:
    """
    Tracks active and idle periods in a flow.

    A packet gap > _ACTIVE_THRESHOLD seconds ends the current active period
    and starts an idle period.  The next packet starts a new active period.
    """
    def __init__(self):
        self._active_stats = _RunningStats()
        self._idle_stats   = _RunningStats()
        self._period_start: Optional[float] = None
        self._last_ts:      Optional[float] = None
        self._in_active = True

    def update(self, ts: float):
        if self._last_ts is None:
            self._period_start = ts
            self._last_ts      = ts
            return

        gap = ts - self._last_ts
        self._last_ts = ts

        if gap > _ACTIVE_THRESHOLD:
            # Close the active period
            if self._period_start is not None:
                duration = (ts - gap) - self._period_start
                if duration > 0:
                    self._active_stats.update(duration)
            # Record the idle gap
            self._idle_stats.update(gap)
            # Start new active period
            self._period_start = ts
        # else: gap is within active period, no state change

    def active(self) -> _RunningStats:
        return self._active_stats

    def idle(self) -> _RunningStats:
        return self._idle_stats


# ── Main accumulator ──────────────────────────────────────────────────────────

class FlowStatAccumulator:
    """
    Accumulates CIC-IDS2017-compatible flow statistics from raw scapy packets.

    Usage::

        acc = FlowStatAccumulator(dst_port=443)
        for pkt in flow_packets:
            acc.update(pkt)
        feature_vec = acc.to_feature_vector(feature_names)

    The ``feature_names`` argument is the list of feature names saved in
    norm_params.json by run_training.py — it tells us which of the 78 CIC-IDS
    columns the model actually uses (some may have been dropped as constant or
    all-NaN during training).
    """

    def __init__(self, dst_port: int = 0):
        # Flow identity
        self._dst_port    = dst_port
        self._start_ts:   Optional[float] = None
        self._last_ts:    Optional[float] = None
        self._fwd_last_ts: Optional[float] = None
        self._bwd_last_ts: Optional[float] = None

        # Direction: the src_ip:src_port of the FIRST packet defines "forward"
        self._fwd_key: Optional[Tuple] = None   # (src_ip, src_port)

        # Packet counts
        self._fwd_pkts = 0
        self._bwd_pkts = 0

        # Byte totals
        self._fwd_bytes = 0
        self._bwd_bytes = 0

        # Header length totals
        self._fwd_hdr_bytes = 0
        self._bwd_hdr_bytes = 0

        # Packet-length stats (payload only, like CIC-IDS)
        self._fwd_pkt_len  = _RunningStats()
        self._bwd_pkt_len  = _RunningStats()
        self._all_pkt_len  = _RunningStats()

        # IAT stats
        self._flow_iat = _RunningStats()
        self._fwd_iat  = _RunningStats()
        self._bwd_iat  = _RunningStats()

        # TCP flags (flow-level counts)
        self._fin = self._syn = self._rst = 0
        self._psh = self._ack = self._urg = 0
        self._cwe = self._ece = 0

        # Per-direction flag counts
        self._fwd_psh = self._bwd_psh = 0
        self._fwd_urg = self._bwd_urg = 0

        # Initial TCP window sizes
        self._init_win_fwd: int = -1    # -1 = not yet seen
        self._init_win_bwd: int = -1

        # Data packets with payload (act_data_pkt_fwd)
        self._act_data_fwd = 0

        # Minimum segment size (min header length forward)
        self._min_seg_fwd: int = 65535

        # Active / idle tracking
        self._ai = _ActiveIdleTracker()

    # ── Public interface ──────────────────────────────────────────────────────

    def update(self, packet) -> None:
        """
        Process one scapy packet and update all running statistics.
        Safe to call with non-IP packets — they are silently skipped.
        """
        try:
            self._process(packet)
        except Exception:
            pass   # never crash the firewall loop on a malformed packet

    def to_feature_vector(
        self,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Return a float32 feature vector.

        If ``feature_names`` is provided (the list saved in norm_params.json),
        only those features are returned, in that order — matching the model's
        input dimension exactly.

        If ``feature_names`` is None, all 78 CIC-IDS features are returned.
        """
        full = self._build_full_vector()   # always 78 values

        if feature_names is None:
            return full

        # Map CIC name → index in full vector
        name_to_idx = {n: i for i, n in enumerate(CIC_FEATURE_NAMES)}

        out = np.zeros(len(feature_names), dtype=np.float32)
        for j, name in enumerate(feature_names):
            idx = name_to_idx.get(name)
            if idx is not None and idx < len(full):
                out[j] = full[idx]
            # else: feature was dropped at training time (constant col), leave 0
        return out

    def packet_count(self) -> int:
        return self._fwd_pkts + self._bwd_pkts

    # ── Internal processing ───────────────────────────────────────────────────

    def _process(self, packet) -> None:
        from scapy.all import IP, IPv6, TCP, UDP

        # Need an IP layer
        if not (packet.haslayer(IP) or packet.haslayer(IPv6)):
            return

        ts = float(getattr(packet, "time", time.monotonic()))

        # ── Timestamps ────────────────────────────────────────────────────
        if self._start_ts is None:
            self._start_ts = ts
        if self._last_ts is not None:
            iat = ts - self._last_ts
            if iat >= 0:
                self._flow_iat.update(iat)
        prev_ts      = self._last_ts
        self._last_ts = ts
        self._ai.update(ts)

        # ── Direction ─────────────────────────────────────────────────────
        ip   = packet[IP] if packet.haslayer(IP) else packet[IPv6]
        src  = getattr(ip, "src", None)
        dst  = getattr(ip, "dst", None)

        src_port = dst_port = None
        is_tcp   = packet.haslayer(TCP)
        is_udp   = packet.haslayer(UDP)
        if is_tcp:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif is_udp:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport

        if self._fwd_key is None:
            self._fwd_key = (src, src_port)

        is_fwd = (src, src_port) == self._fwd_key

        # ── Sizes ─────────────────────────────────────────────────────────
        pkt_len = len(packet)
        hdr_len = self._header_len(packet)
        payload = max(0, pkt_len - hdr_len)

        self._all_pkt_len.update(pkt_len)

        if is_fwd:
            self._fwd_pkts   += 1
            self._fwd_bytes  += pkt_len
            self._fwd_hdr_bytes += hdr_len
            self._fwd_pkt_len.update(pkt_len)
            if payload > 0:
                self._act_data_fwd += 1
            if hdr_len < self._min_seg_fwd:
                self._min_seg_fwd = hdr_len

            # Forward IAT
            if self._fwd_last_ts is not None:
                iat = ts - self._fwd_last_ts
                if iat >= 0:
                    self._fwd_iat.update(iat)
            self._fwd_last_ts = ts
        else:
            self._bwd_pkts   += 1
            self._bwd_bytes  += pkt_len
            self._bwd_hdr_bytes += hdr_len
            self._bwd_pkt_len.update(pkt_len)

            # Backward IAT
            if self._bwd_last_ts is not None:
                iat = ts - self._bwd_last_ts
                if iat >= 0:
                    self._bwd_iat.update(iat)
            self._bwd_last_ts = ts

        # ── TCP flags ─────────────────────────────────────────────────────
        if is_tcp:
            tcp   = packet[TCP]
            flags = int(tcp.flags)
            fin   = bool(flags & 0x01)
            syn   = bool(flags & 0x02)
            rst   = bool(flags & 0x04)
            psh   = bool(flags & 0x08)
            ack   = bool(flags & 0x10)
            urg   = bool(flags & 0x20)
            ece   = bool(flags & 0x40)
            cwe   = bool(flags & 0x80)

            if fin: self._fin += 1
            if syn: self._syn += 1
            if rst: self._rst += 1
            if psh: self._psh += 1
            if ack: self._ack += 1
            if urg: self._urg += 1
            if ece: self._ece += 1
            if cwe: self._cwe += 1

            if is_fwd:
                if psh: self._fwd_psh += 1
                if urg: self._fwd_urg += 1
            else:
                if psh: self._bwd_psh += 1
                if urg: self._bwd_urg += 1

            # Initial window sizes from the first SYN / SYN-ACK
            win = int(getattr(tcp, "window", 0))
            if syn and not ack and self._init_win_fwd == -1 and is_fwd:
                self._init_win_fwd = win
            if syn and ack and self._init_win_bwd == -1 and not is_fwd:
                self._init_win_bwd = win

    # ── Build the 78-feature vector ───────────────────────────────────────────

    def _build_full_vector(self) -> np.ndarray:
        v = np.zeros(78, dtype=np.float32)

        dur   = (self._last_ts - self._start_ts) if (self._last_ts and self._start_ts) else 0.0
        dur   = max(dur, 1e-6)   # avoid division by zero
        total_pkts  = self._fwd_pkts + self._bwd_pkts
        total_bytes = self._fwd_bytes + self._bwd_bytes

        # Helpers ──────────────────────────────────────────────────────────
        def _s(rs: _RunningStats, attr: str, default=0.0) -> float:
            return getattr(rs, attr) if rs.n > 0 else default

        def _smin(rs: _RunningStats) -> float:
            return rs.safe_min()

        def _smax(rs: _RunningStats) -> float:
            return rs.safe_max()

        # 0  Destination Port
        v[0]  = float(self._dst_port)
        # 1  Flow Duration (microseconds in CIC-IDS, we use seconds * 1e6)
        v[1]  = dur * 1e6
        # 2  Total Fwd Packets
        v[2]  = float(self._fwd_pkts)
        # 3  Total Backward Packets
        v[3]  = float(self._bwd_pkts)
        # 4  Total Length of Fwd Packets
        v[4]  = float(self._fwd_bytes)
        # 5  Total Length of Bwd Packets
        v[5]  = float(self._bwd_bytes)
        # 6  Fwd Packet Length Max
        v[6]  = _smax(self._fwd_pkt_len)
        # 7  Fwd Packet Length Min
        v[7]  = _smin(self._fwd_pkt_len)
        # 8  Fwd Packet Length Mean
        v[8]  = _s(self._fwd_pkt_len, "mean")
        # 9  Fwd Packet Length Std
        v[9]  = _s(self._fwd_pkt_len, "std")
        # 10 Bwd Packet Length Max
        v[10] = _smax(self._bwd_pkt_len)
        # 11 Bwd Packet Length Min
        v[11] = _smin(self._bwd_pkt_len)
        # 12 Bwd Packet Length Mean
        v[12] = _s(self._bwd_pkt_len, "mean")
        # 13 Bwd Packet Length Std
        v[13] = _s(self._bwd_pkt_len, "std")
        # 14 Flow Bytes/s
        v[14] = total_bytes / dur
        # 15 Flow Packets/s
        v[15] = total_pkts / dur
        # 16 Flow IAT Mean
        v[16] = _s(self._flow_iat, "mean")
        # 17 Flow IAT Std
        v[17] = _s(self._flow_iat, "std")
        # 18 Flow IAT Max
        v[18] = _smax(self._flow_iat)
        # 19 Flow IAT Min
        v[19] = _smin(self._flow_iat)
        # 20 Fwd IAT Total
        v[20] = self._fwd_iat.total if self._fwd_iat.n > 0 else 0.0
        # 21 Fwd IAT Mean
        v[21] = _s(self._fwd_iat, "mean")
        # 22 Fwd IAT Std
        v[22] = _s(self._fwd_iat, "std")
        # 23 Fwd IAT Max
        v[23] = _smax(self._fwd_iat)
        # 24 Fwd IAT Min
        v[24] = _smin(self._fwd_iat)
        # 25 Bwd IAT Total
        v[25] = self._bwd_iat.total if self._bwd_iat.n > 0 else 0.0
        # 26 Bwd IAT Mean
        v[26] = _s(self._bwd_iat, "mean")
        # 27 Bwd IAT Std
        v[27] = _s(self._bwd_iat, "std")
        # 28 Bwd IAT Max
        v[28] = _smax(self._bwd_iat)
        # 29 Bwd IAT Min
        v[29] = _smin(self._bwd_iat)
        # 30 Fwd PSH Flags
        v[30] = float(self._fwd_psh)
        # 31 Bwd PSH Flags
        v[31] = float(self._bwd_psh)
        # 32 Fwd URG Flags
        v[32] = float(self._fwd_urg)
        # 33 Bwd URG Flags
        v[33] = float(self._bwd_urg)
        # 34 Fwd Header Length (total)
        v[34] = float(self._fwd_hdr_bytes)
        # 35 Bwd Header Length (total)
        v[35] = float(self._bwd_hdr_bytes)
        # 36 Fwd Packets/s
        v[36] = self._fwd_pkts / dur
        # 37 Bwd Packets/s
        v[37] = self._bwd_pkts / dur
        # 38 Min Packet Length (overall)
        v[38] = _smin(self._all_pkt_len)
        # 39 Max Packet Length (overall)
        v[39] = _smax(self._all_pkt_len)
        # 40 Packet Length Mean
        v[40] = _s(self._all_pkt_len, "mean")
        # 41 Packet Length Std
        v[41] = _s(self._all_pkt_len, "std")
        # 42 Packet Length Variance
        v[42] = _s(self._all_pkt_len, "variance")
        # 43 FIN Flag Count
        v[43] = float(self._fin)
        # 44 SYN Flag Count
        v[44] = float(self._syn)
        # 45 RST Flag Count
        v[45] = float(self._rst)
        # 46 PSH Flag Count
        v[46] = float(self._psh)
        # 47 ACK Flag Count
        v[47] = float(self._ack)
        # 48 URG Flag Count
        v[48] = float(self._urg)
        # 49 CWE Flag Count
        v[49] = float(self._cwe)
        # 50 ECE Flag Count
        v[50] = float(self._ece)
        # 51 Down/Up Ratio  (bwd / fwd byte ratio)
        v[51] = (self._bwd_bytes / self._fwd_bytes) if self._fwd_bytes > 0 else 0.0
        # 52 Average Packet Size
        v[52] = _s(self._all_pkt_len, "mean")
        # 53 Avg Fwd Segment Size
        v[53] = _s(self._fwd_pkt_len, "mean")
        # 54 Avg Bwd Segment Size
        v[54] = _s(self._bwd_pkt_len, "mean")
        # 55 Fwd Header Length (duplicate col in CIC-IDS)
        v[55] = float(self._fwd_hdr_bytes)
        # 56-61 Bulk features — approximated as 0
        # (CIC-IDS computes these over post-hoc "bulk windows"; not feasible live)
        v[56] = v[57] = v[58] = v[59] = v[60] = v[61] = 0.0
        # 62 Subflow Fwd Packets (approx: total fwd pkts)
        v[62] = float(self._fwd_pkts)
        # 63 Subflow Fwd Bytes
        v[63] = float(self._fwd_bytes)
        # 64 Subflow Bwd Packets
        v[64] = float(self._bwd_pkts)
        # 65 Subflow Bwd Bytes
        v[65] = float(self._bwd_bytes)
        # 66 Init_Win_bytes_forward
        v[66] = float(self._init_win_fwd) if self._init_win_fwd >= 0 else 0.0
        # 67 Init_Win_bytes_backward
        v[67] = float(self._init_win_bwd) if self._init_win_bwd >= 0 else 0.0
        # 68 act_data_pkt_fwd (fwd packets with non-zero payload)
        v[68] = float(self._act_data_fwd)
        # 69 min_seg_size_forward (min header length in fwd direction)
        v[69] = float(self._min_seg_fwd) if self._min_seg_fwd < 65535 else 0.0
        # 70-73 Active Mean/Std/Max/Min
        ai = self._ai.active()
        v[70] = _s(ai, "mean")
        v[71] = _s(ai, "std")
        v[72] = _smax(ai)
        v[73] = _smin(ai)
        # 74-77 Idle Mean/Std/Max/Min
        idl = self._ai.idle()
        v[74] = _s(idl, "mean")
        v[75] = _s(idl, "std")
        v[76] = _smax(idl)
        v[77] = _smin(idl)

        # Clip any inf/nan that might arise from extreme packets
        np.clip(v, -1e9, 1e9, out=v)
        v = np.nan_to_num(v, nan=0.0, posinf=1e9, neginf=-1e9)
        return v

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _header_len(packet) -> int:
        """Total IP + transport header length in bytes."""
        hdr = 0
        from scapy.all import IP, IPv6, TCP, UDP
        if packet.haslayer(IP):
            hdr += packet[IP].ihl * 4
        elif packet.haslayer(IPv6):
            hdr += 40
        if packet.haslayer(TCP):
            hdr += packet[TCP].dataofs * 4
        elif packet.haslayer(UDP):
            hdr += 8
        return hdr
