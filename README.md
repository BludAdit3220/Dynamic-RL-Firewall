## Dynamic Reinforcement Learning Firewall (Python)

This repository contains a **next‑generation adaptive firewall prototype** built in Python.  
Instead of using only static rules, it trains a **Deep Q‑Network (DQN)** to learn a policy that:

- **Allows benign traffic**
- **Blocks or rate‑limits malicious traffic**
- **Penalizes false positives and false negatives**

The project is designed as a **research / graduation‑level project** showing how **AI + cybersecurity** can be combined into a **self‑improving, adaptive firewall**.

It uses:

- **OpenAI Gymnasium** (`DynamicFirewallEnv`) for the RL environment
- **TensorFlow / Keras** (`DQNAgent`) for the policy network
- **iptables + scapy** for packet handling and (optional) live packet filtering
- **Streamlit** as a dashboard for traffic statistics and live policy decisions

---

### 1. Project Structure

- `firewall_env.py` — Gymnasium environment over preprocessed traffic features and labels
- `agent_dqn.py` — DQN agent implementation using Keras
- `run_training.py` — offline training on labeled traffic dataset(s)
- `run_firewall.py` — live firewall loop: sniff packets, run RL policy, optionally apply iptables rules, and log decisions
- `iptables_manager.py` — helper for constructing and applying iptables rules (safe by default via `dry_run=True`)
- `evaluate_vs_ruleset.py` — compare RL firewall vs. a simple static ruleset (detection rate, FPR, accuracy)
- `dashboard.py` — Streamlit dashboard visualizing traffic and decisions from log file
- `data/` — offline CSV datasets (e.g. CIC‑IDS fragments like `Monday-WorkingHours.pcap_ISCX.csv`, etc.)
- `models/` — saved DQN models
- `logs/` — firewall decision logs used by the dashboard
- `requirements.txt` — Python dependencies

---

### 2. What the RL Firewall Is Doing

- **Offline phase (training / evaluation)**  
  - You provide labeled traffic in CSV form.  
  - `run_training.py` wraps it in `DynamicFirewallEnv`, which:
    - Feeds feature vectors as observations to the agent
    - Gives positive reward for:
      - Allowing benign traffic
      - Blocking malicious traffic
    - Applies negative reward for:
      - Blocking benign traffic (false positives)
      - Allowing malicious traffic (false negatives)
      - Mild penalty for rate‑limiting (latency cost)
  - The `DQNAgent` learns a Q‑function over actions `allow / block / rate‑limit`.

- **Online phase (live firewall)**  
  - `run_firewall.py`:
    - Uses `scapy` to sniff live packets from an interface
    - Extracts simple features from each packet
    - Uses the trained DQN model to choose an action
    - Optionally converts decisions into iptables rules (in non‑dry‑run mode)
    - Logs every decision to a CSV file for later analysis and visualization.

- **Visualization and comparison**  
  - `dashboard.py` reads the log file and gives you a live view of decisions and traffic stats.
  - `evaluate_vs_ruleset.py` compares the learned DQN policy to a simple, static baseline ruleset.

---

### 3. Data Format and How to Use Multiple CSVs

All offline training/evaluation scripts expect CSVs with this basic format:

```text
f1,f2,...,fN,label
0.1,0.2,...,1.3,0
0.5,0.9,...,0.3,1
...
```

- **Features**: `f1..fN` — numeric features extracted from packets/flows (length, ports, flow rates, entropy, etc.)
- **Label**: `label` — `0` = benign, `1` = malicious  
  (You can extend to multi‑class if you also update the reward logic.)

You can build these CSVs from:

- PCAPs (e.g. CIC‑IDS dataset) using `scapy` or `tshark` exports
- Logs from an IDS/IPS such as Snort or Suricata

#### 3.1 Using a Single CSV

You can still train on a single combined CSV:

- Example: `data/train_traffic.csv`

#### 3.2 Using Many CSV Files (e.g. all CIC‑IDS splits)

This project has been updated so that **both** `run_training.py` and `evaluate_vs_ruleset.py` can take either:

- **A single CSV file**, or
- **A directory containing many CSVs**

If you point `--dataset` at a **directory** (for example, `data/` that contains:

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

then the scripts will:

- Find all `*.csv` files in that directory
- Read each CSV with pandas
- Concatenate them into one big dataset in memory
- Split into features and labels exactly as before

This lets you train on **the whole CIC‑IDS period at once** without manually merging files.

---

### 4. Setup and Installation

Recommended: use Python 3.10+ in a virtual environment.

```bash
cd dynamic_rl_firewall
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5. Training the RL Firewall

#### 5.1 Training on a Single CSV

```bash
cd dynamic_rl_firewall

python run_training.py \
  --dataset data/train_traffic.csv \
  --out models/rl_firewall_dqn \
  --episodes 100
```

#### 5.2 Training on All CSVs in `data/` (e.g. full CIC‑IDS)

Place all your CSVs under `data/` and run:

```bash
cd dynamic_rl_firewall

python run_training.py \
  --dataset data \
  --out models/rl_firewall_dqn \
  --episodes 100
```

Internally, `run_training.py` will detect that `data` is a directory, load **all** the `*.csv` files, concatenate them, and then train on the combined dataset.

The script uses `DynamicFirewallEnv` to:

- Present feature vectors as observations
- Reward **allowing benign** traffic and **blocking malicious** traffic
- Penalize **false positives and false negatives**
- Apply a small additional penalty for rate‑limiting

The trained Keras model is saved under `models/rl_firewall_dqn`.

---

### 6. Comparing RL Firewall vs Traditional Rules

Use the evaluation script to compare the learned DQN firewall to a simple static baseline ruleset.

#### 6.1 Evaluate on a Single CSV

```bash
cd dynamic_rl_firewall

python evaluate_vs_ruleset.py \
  --model models/rl_firewall_dqn \
  --dataset data/test_traffic.csv
```

#### 6.2 Evaluate on All CSVs in `data/`

```bash
cd dynamic_rl_firewall

python evaluate_vs_ruleset.py \
  --model models/rl_firewall_dqn \
  --dataset data
```

The script reports:

- **Detection rate** (TPR)
- **False positive rate**
- **Overall accuracy**

for:

- The **RL firewall** (DQN policy)
- A **static heuristic ruleset** (hand‑crafted baseline)

You can extend `baseline_rule_engine` with more realistic rules or export rules from an existing firewall for more realistic comparisons.

---

### 7. Running the Live Firewall (Linux)

> **Safety First:** By default, the live loop runs in **dry‑run mode** and does *not* modify iptables.  
> Only enable `--apply` after testing carefully and **only** if you have console access to recover from misconfigurations.

Run the live decision loop:

```bash
cd dynamic_rl_firewall

sudo python run_firewall.py \
  --model models/rl_firewall_dqn \
  --interface eth0 \
  --log logs/firewall_events.csv
```

- Uses `scapy.sniff` to capture packets from the given interface
- Builds a simple feature vector from each packet
- Uses the DQN policy to pick an action: `allow` / `block` / `rate‑limit`
- Logs every decision to `logs/firewall_events.csv`

To actually push **iptables rules**, add `--apply`:

```bash
cd dynamic_rl_firewall

sudo python run_firewall.py \
  --model models/rl_firewall_dqn \
  --interface eth0 \
  --log logs/firewall_events.csv \
  --apply
```

`iptables_manager.py` translates decisions into `Rule` objects and (if `dry_run=False`) invokes the corresponding `iptables` commands.

---

### 8. Dashboard: Traffic & Policy Visualization

Run the Streamlit dashboard:

```bash
cd dynamic_rl_firewall
streamlit run dashboard.py
```

The dashboard shows:

- **Recent decisions** with src/dst IPs, ports, and actions
- **Aggregated actions over time** (line chart)
- **Top talkers** (top source IPs by event count)
- High‑level metrics (allowed vs blocked/rate‑limited)

By default it points at `logs/firewall_events.csv`, which is generated by `run_firewall.py`.  
You can change the log path from the sidebar.

---

### 9. Extending the Research / Graduation Project

Ideas to deepen or customize the project:

- **Algorithm**: replace DQN with PPO / A2C for potentially more stable learning
- **Features**: enrich features with **flow‑level context** and temporal windows
- **Labels**: integrate labels from an external IDS (e.g., Suricata) as delayed rewards
- **Self‑healing**: automatically roll back policies that cause bursts of false positives
- **Visualization**: extend the dashboard to show **policy drift** and **reward trends** over time

The codebase is intentionally modular so you can iterate on:

- The ML model
- Feature engineering
- Rule engine

independently, while still demonstrating a full **dynamic, self‑improving firewall pipeline**.
