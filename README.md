#  Dynamic RL Firewall — LSTM-DQN Network Intrusion Detection System

> A real-time network intrusion detection system using a **Dueling LSTM Deep Q-Network** that makes block/allow/monitor decisions on live network flows with sequential temporal memory — something impossible with stateless classifiers.

<p align="center">
  <img src="assets/architecture.png" alt="LSTM-DQN Architecture" width="800"/>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [How It Works — The RL Formulation](#-how-it-works--the-rl-formulation)
- [Results](#-results)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Live Firewall](#-live-firewall)
- [Limitations & Honest Assessment](#-limitations--honest-assessment)
- [Tech Stack](#-tech-stack)

---

##  Overview

Traditional firewalls rely on static rules — port numbers, IP blacklists, protocol flags. They have no memory, no adaptability, and no concept of *how a connection behaves over time*.

This project replaces static rules with a **Reinforcement Learning agent** that:

- **Observes** network flow statistics packet-by-packet as a time sequence
- **Remembers** previous packets via LSTM hidden state
- **Decides** in real-time: `BLOCK`, `ALLOW`, or `MONITOR` each connection
- **Learns** from rewards and penalties tied to detection accuracy
- **Improves** its policy over thousands of training episodes

The key insight: **detecting DDoS after 3 packets is more valuable than detecting it after 10.** The LSTM enables exactly this — early blocking using sequential context.

---

##  Architecture

### Dueling LSTM Deep Q-Network

```
Input (70 CIC-IDS features per packet)
        │
   ┌────▼─────┐
   │ FC 128   │  ← Encoder: compress raw features
   │  ReLU    │
   └────┬─────┘
        │
   ┌────▼──────────────┐
   │   LSTM (256)      │  ← Temporal memory across packet sequence
   │   h_t, c_t  ←────┘     hidden state carries context forward
   └────┬──────────────┘
        │
   ┌────┴──────────────────────────┐
   │                               │
┌──▼──────┐                  ┌────▼──────┐
│ FC 128  │                  │  FC 128   │  ← Dueling heads
│ Value   │                  │ Advantage │
│  V(s)   │                  │  A(s,a)   │
└──┬──────┘                  └────┬──────┘
   │                               │
   └──────────┬────────────────────┘
              │
   Q(s,a) = V(s) + A(s,a) − mean(A)
              │
   ┌──────────┼──────────┐
   ▼          ▼          ▼
 BLOCK      ALLOW     MONITOR
```

### Why Each Component

| Component | Purpose |
|---|---|
| **LSTM** | Temporal context — detects attack *patterns* across packets, not just individual packet stats |
| **Dueling heads** | Separates "how dangerous is this flow state" (V) from "which action is best" (A) — more stable Q-learning |
| **70 CIC-IDS features** | Flow-level stats: packet rates, byte counts, inter-arrival times, flag counts (same as real IDS systems) |
| **3 actions** | `BLOCK` (drop + iptables rule), `ALLOW` (pass), `MONITOR` (log only, non-destructive) |

---

##  How It Works — The RL Formulation

### Markov Decision Process

| MDP Element | Definition |
|---|---|
| **State** `s` | 70-dim CIC-IDS flow features at timestep `t`, plus LSTM hidden state `(h_t, c_t)` |
| **Action** `a` | `{0: BLOCK, 1: ALLOW, 2: MONITOR}` |
| **Reward** `r` | Shaped per-packet signal (see below) |
| **Discount** `γ` | 0.99 — agent values future correct decisions |
| **Episode** | One 10-packet network flow sequence |

### Reward Function

```python
# Attack packet
if is_attack:
    reward = +5.0   if action == BLOCK   else -10.0  # catch it or miss it
# Benign packet
else:
    reward = +1.0   if action == ALLOW   else -5.0   # pass it or false alarm
# Monitor action
reward += 0.5  # small bonus for non-destructive monitoring
```

The asymmetric penalty (missing an attack costs 2× more than a false positive) teaches the agent to be security-first.

### Training Algorithm: Dueling DQN

```
1. Collect experience: (s_t, a_t, r_t, s_{t+1}) → Replay Buffer
2. Sample random mini-batch from buffer
3. Compute target: y = r + γ · max_a Q_target(s', a)
4. Minimise: L = (y − Q_online(s, a))²
5. Soft-update target network: θ_target ← τ·θ + (1−τ)·θ_target
6. Decay ε for exploration → exploitation transition
```

**Key training decisions:**
- **Replay buffer**: breaks temporal correlation — prevents LSTM from overfitting to sequential patterns
- **Target network** (`τ=0.005` soft update): stabilises Q-targets, prevents divergence
- **ε-greedy** decay from 1.0→0.05: explores all attack types before committing to a policy
- **Minority oversampling**: attack episodes stored 2× in buffer — corrects 80:20 class imbalance
- **Patience stops early**: halts if validation score stops improving after agent is in exploitation mode

---

##  Results

### Performance Across All CIC-IDS2017 Attack Types

| Attack Type | Flow DR | Flow FPR | Early DR | Youden's J |
|---|---|---|---|---|
| 🟢 **DDoS** | **100.0%** | 11.78% | **100.0%** | **+0.882** |
| 🟢 **PortScan** | **100.0%** | 17.81% | **100.0%** | **+0.822** |
| 🟢 **DoS (Hulk, GoldenEye, slowloris, Slowhttptest, Heartbleed)** | **99.91%** | 16.45% | **99.81%** | **+0.835** |
| 🟡 **Bot** | 89.34% | 16.41% | 88.32% | +0.729 |
| 🟡 **FTP-Patator / SSH-Patator** | 67.12% | 17.78% | 64.31% | +0.493 |
| 🔴 **Web Attacks (SQLi, XSS, Brute Force)** | 11.93% | 15.67% | 11.93% | −0.037 |
| ⚪ **Infiltration** | 0.0% | — | — | — *(only 4 labelled flows — not evaluable)* |

> **Youden's J = Detection Rate − False Positive Rate.** Score of +1.0 is perfect; 0.0 is random.

### RL Agent vs Static Ruleset — DDoS Head-to-Head

| Metric | LSTM-DQN | Static Rules* | Improvement |
|---|---|---|---|
| Detection Rate | **100.0%** | 99.7% | Matched (both excellent) |
| False Positive Rate | **1.33%** | 58.0% | **43× fewer false alarms** |
| F1-Score | **0.995** | 0.810 | **+0.185** |
| MCC | **0.988** | 0.530 | Near-perfect vs weak |

> *\* The static baseline was upgraded to use 8 real-world heuristic rules (volumetric floods, high-variance bursts, zero-payload scans, etc.) based on raw CIC-IDS features. While it successfully detects 99.7% of DDoS attacks, it suffers from a massive 58% false positive rate because rigid thresholds cannot separate complex attack traffic from heavy legitimate usage.*

### What "Early Detection Rate" Proves

```
Full-flow DR  (10 packets seen): 98.68%
Early DR      ( 5 packets seen): 98.56%   ← only 0.12% worse!
```

The model makes the same correct decision halfway through a connection as it does at the end. This is the LSTM's hidden state doing temporal credit assignment — **a stateless classifier cannot replicate this**.

### Combined Validation (All Attack Types, Best Checkpoint)

```
Flow Detection Rate  : 98.68%
Flow False Pos. Rate : 20.27%
Early Detection Rate : 98.56%
TP=8,252  TN=27,184  FP=6,912  FN=110
```

---

##  Dataset

**CIC-IDS2017** — Canadian Institute for Cybersecurity Intrusion Detection dataset

| File | Attack Types | Size |
|---|---|---|
| Monday | BENIGN only | 169 MB |
| Tuesday | FTP-Patator, SSH-Patator | 129 MB |
| Wednesday | DoS Hulk/GoldenEye/Slowhttptest/slowloris, Heartbleed | 215 MB |
| Thursday AM | Web Attack Brute Force, XSS, SQL Injection | 50 MB |
| Thursday PM | Infiltration | 80 MB |
| Friday AM | PortScan, Bot | 56 MB |
| Friday PM | DDoS, PortScan | 148 MB |

**Total**: 2,830,743 rows — 2,273,097 benign, 557,646 attacks (19.7% attack ratio)

> Download from: https://www.unb.ca/cic/datasets/ids-2017.html  
> Place all CSVs inside the `data/` directory.

---

##  Project Structure

```
Dynamic-RL-Firewall/
├── data/                         # CIC-IDS2017 CSV files (not in repo)
├── models/
│   ├── rl_firewall_lstm.pt       # Trained LSTM-DQN weights
│   └── norm_params.json          # Feature mean/std + names (saved at train time)
├── assets/
│   └── architecture.png          # LSTM-DQN architecture diagram
│
├── run_training.py               # Main training script (RL loop)
├── evaluate_vs_ruleset.py        # Evaluation: RL vs static baseline
├── run_firewall.py               # Live firewall using scapy + iptables
│
├── agent_dqn.py                  # LSTMDQNAgent, LSTMQNetwork, ReplayBuffer
├── firewall_env.py               # FlowFirewallEnv (OpenAI Gym-compatible MDP)
├── flow_builder.py               # Groups packets into flow episodes
├── flow_stats.py                 # Online Welford stats accumulator (live inference)
├── load_dataset.py               # CIC-IDS CSV loader + feature cleaning
│
└── requirements.txt
```

---

##  Installation

```bash
git clone https://github.com/BludAdit3220/Dynamic-RL-Firewall
cd Dynamic-RL-Firewall

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

##  Training

```bash
python run_training.py \
    --dataset data/ \
    --out models/rl_firewall_lstm.pt \
    --episodes 500 \
    --flow-len 10 \
    --seq-len 8 \
    --patience 30 \
    --min-buffer 200 \
    --val-every 5
```

| Argument | Default | Description |
|---|---|---|
| `--dataset` | — | Path to folder containing CIC-IDS CSVs |
| `--out` | `models/rl_firewall_lstm.pt` | Where to save the model |
| `--episodes` | 500 | Total RL training episodes |
| `--flow-len` | 10 | Packets per flow episode |
| `--seq-len` | 8 | LSTM lookback window |
| `--patience` | 30 | Val checks without improvement before early stop |
| `--min-buffer` | 200 | Episodes to collect before first gradient update |
| `--val-every` | 5 | Run full validation every N episodes |

### What to Watch During Training

```
Ep  171/500 | type=DDoS | loss=2.5511 | ε=1.000 → buffer warm, first gradient
Ep  216/500 | type=... | score=+0.819 | ε=0.998 → model improving
Ep  350/500 | type=... | ε=0.390     → ε < 0.4, model saved to disk ✓
Ep  440/500 | type=... | ε=0.050     → fully greedy, best checkpoint locked
```

---

##  Evaluation

Evaluate on any single CIC-IDS2017 CSV (automatic feature alignment to training set):

```bash
python evaluate_vs_ruleset.py \
    --model models/rl_firewall_lstm.pt \
    --dataset data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \
    --flow-len 10
```

Outputs:
- Packet-level and flow-level confusion matrices
- DR, FPR, F1, MCC, Youden's J for RL agent
- Same metrics for static rule baseline
- Early Detection Rate (the LSTM temporal signal)

---

##  Live Firewall

> ⚠️ **Requires root / sudo. Adds iptables rules. Use in a VM or test environment.**

```bash
sudo python run_firewall.py \
    --model models/rl_firewall_lstm.pt \
    --iface eth0 \
    --dry-run          # remove --dry-run to actually block traffic
```

**How live inference works:**

```
Network packet arrives
        │
  scapy sniff()
        │
  FlowStatAccumulator  ← Welford online stats per flow (no raw PCAP needed)
        │
  70-feature vector (normalised with saved mean/std)
        │
  LSTM-DQN forward pass  (maintains h_t, c_t per flow)
        │
  argmax Q-value
        │
  ┌─────┼─────┐
BLOCK ALLOW MONITOR
  │
iptables -A INPUT -s <ip> -j DROP
```

---

##  Limitations & Honest Assessment

### What This Model Cannot Detect Well

| Attack Class | Detection | Root Cause |
|---|---|---|
| **Web Attacks** (SQLi, XSS) | ~12% | Attack lives in HTTP payload — flow stats are identical to normal browsing. Requires Deep Packet Inspection at Layer 7. |
| **Infiltration** | Not evaluable | CIC-IDS2017 has only 4 labelled infiltration flows — a data quality issue in the dataset, not the model. |
| **Credential brute-force** | ~67% | Rate-limited variants look like normal authentication. Would improve with IP-level session counting. |

### Architectural Ceiling

The CIC-IDS2017 features are **flow-level statistics** (rates, counts, durations). They do not include packet payloads. This means:

-  Volumetric attacks (DDoS, DoS, PortScan) — detectable
-  Behavioural anomalies (Bot, unusual flow patterns) — partially detectable
-  Application-layer semantic attacks (SQLi, XSS) — require DPI, out of scope

This is a property of the feature set, not the RL algorithm.

---

##  Tech Stack

| Component | Technology |
|---|---|
| **RL Algorithm** | Dueling DQN with LSTM (PyTorch) |
| **Temporal Memory** | LSTM (hidden=256, seq_len=8) |
| **Training Stability** | Experience Replay, Soft Target Network (τ=0.005) |
| **Online Stats** | Welford's algorithm (O(1) per packet) |
| **Live Packet Capture** | Scapy |
| **Firewall Integration** | iptables via `python-iptables` |
| **Dataset** | CIC-IDS2017 (2.83M flows, 15 attack types) |
| **Language** | Python 3.10+ |

---

##  References

- Wang, W. et al. (2017). *HAST-IDS: Learning Hierarchical Spatial-Temporal Features Using Deep Neural Networks.* — LSTM for IDS inspiration
- Wang, Z. et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning.* ICML. — Dueling DQN
- Sharafaldin, I. et al. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICAISS. — CIC-IDS2017 dataset
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature. — Original DQN

---

<p align="center">
  Built with PyTorch · CIC-IDS2017 · Dueling LSTM-DQN · iptables
</p>
