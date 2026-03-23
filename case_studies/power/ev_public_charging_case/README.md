# EV Public Charging Case Study

A multi-agent, hierarchical EV charging station environment built on HERON (Hierarchical Reinforcement Learning Framework). This case study demonstrates real-world multi-agent coordination for electric vehicle charging pricing and power management.

## Overview

This environment simulates a network of public EV charging stations where:

- **Multiple Stations** operate autonomously, each managing a fixed pool of charger slots
- **Pricing Coordination** uses reinforcement learning to optimize dynamic pricing strategies
- **EV Arrivals & Charging Dynamics** follow realistic patterns with customer price sensitivity
- **Market Integration** receives real-time electricity prices (LMP) from external markets
- **Regulation Compliance** tracks frequency regulation signals and compliance metrics

The implementation follows HERON's **CTDE (Centralized Training, Decentralized Execution)** pattern, enabling both synchronous training and asynchronous event-driven deployment.

## Architecture

### Agent Hierarchy

```
SystemAgent (L3)
  │
  ├─ StationCoordinator₁ (L2) ← Makes pricing decisions
  │  ├─ ChargingSlot₁ (L1)   ← Controls individual charger
  │  ├─ ChargingSlot₂ (L1)
  │  └─ ChargingSlot₃ (L1)
  │
  └─ StationCoordinator₂ (L2)
     ├─ ChargingSlot₁ (L1)
     ├─ ChargingSlot₂ (L1)
     └─ ChargingSlot₃ (L1)
```

### Components

#### Agents

**`ChargingSlot`** (Field Agent)
- Represents a single charger port with optional EV occupancy
- **Features:**
  - `ChargerFeature`: Physical charger state (power, max power, availability)
  - `EVSlotFeature`: EV state (occupied, SOC, arrival time, price sensitivity)
- **Action:** 1D continuous price broadcast from coordinator `[0.0, 0.8]` $/kWh
- **Role:** Receives price from coordinator, determines charging power based on EV state and price sensitivity

**`StationCoordinator`** (Coordinator Agent)
- Manages multiple `ChargingSlot` subordinates
- **Features:**
  - `ChargingStationFeature`: Station-level metrics (available chargers, current price)
  - `MarketFeature`: Market signals (LMP, time-of-day, next period LMP)
  - `RegulationFeature`: Frequency regulation metrics (signal, compliance, status)
- **Observation:** 8D vector
  - Open charger count (normalized)
  - Current price (normalized)
  - Market prices (LMP, forecast)
  - Regulation signal/headroom
- **Action:** 1D continuous pricing `[0.0, 0.8]` $/kWh
- **Role:** Optimizes pricing strategy to maximize revenue while respecting constraints

#### Environment

**`ChargingEnv`** (HeronEnv)
- Extends HERON's `HeronEnv` base class
- Manages multi-station simulation with shared state
- **Responsibilities:**
  - EV arrivals: Poisson process at configurable rate
  - Charging dynamics: Power constrained by slot capacity and EV battery state
  - Departures: EVs leave when SOC reaches target or max wait time exceeded
  - Revenue tracking: Accumulated from charged kWh
  - Reward computation: Multi-objective (revenue, efficiency, fairness)

#### Policies

**`PricingPolicy`** (Actor-Critic)
- Neural network-based policy for dynamic pricing
- **Architecture:**
  - Actor MLP: Input 8D → Hidden layer (32D) → Output 1D action (sigmoid scaled to [0, 0.8])
  - Critic MLP: Input 8D → Hidden layer (32D) → Output 1D value estimate
- **Learning:** Policy gradient with advantage estimation and baseline subtraction
- **Noise decay:** Exploration noise decreases over time for convergence

#### Features

All features are implementations of `Feature` and return normalized vectors:

- **`ChargerFeature`**: Power (p_kw), max power, operational status
- **`EVSlotFeature`**: Occupancy, SOC (state of charge), arrival time, price sensitivity
- **`ChargingStationFeature`**: Available chargers, pricing decision
- **`MarketFeature`**: LMP, time of day, next period forecast
- **`RegulationFeature`**: Regulation signal, up/down headroom, compliance

## Quick Start

### Installation

```bash
# Install PowerGym with dependencies
pip install -e /Users/yulinzeng/PycharmProjects/PowerGym
```

### Basic Environment Usage

```python
from case_studies.power.ev_public_charging_case.agents import ChargingSlot, StationCoordinator
from case_studies.power.ev_public_charging_case.envs.charging_env import ChargingEnv

# Create a station with 5 charger slots
station_id = "station_0"
slots = {
    f"{station_id}_slot_{j}": ChargingSlot(
        agent_id=f"{station_id}_slot_{j}",
        p_max_kw=150.0
    )
    for j in range(5)
}
coordinator = StationCoordinator(agent_id=station_id, subordinates=slots)

# Initialize environment
env = ChargingEnv(
    coordinator_agents=[coordinator],
    arrival_rate=10.0,  # EVs per hour
    dt=300.0,           # 5-minute timesteps
    episode_length=86400.0,  # 24 hours
)

# Run episode
obs, info = env.reset(seed=42)
for step in range(288):  # 288 steps = 24 hours
    actions = {"station_0": np.array([0.3])}  # Price at $0.30/kWh
    obs, rewards, terminated, truncated, infos = env.step(actions)
    if truncated["__all__"]:
        break
```

### Running Provided Scripts

#### 1. Single-Station Rollout

Quick test with random pricing:

```bash
python -m case_studies.power.ev_public_charging_case.run_single_station_rollout
```

Output:
```
Step   0 | Reward:   0.005 | Cumulative:    0.005
Step  50 | Reward:   0.024 | Cumulative:    1.234
...
Total reward over 288 steps: 156.78
```

#### 2. Training with Policy Gradient (CTDE)

Train pricing policies using HERON's centralized training pipeline:

```bash
python -m case_studies.power.ev_public_charging_case.train_rllib
```

**What happens:**
1. Creates 2-station environment (2 slots per station by default)
2. Initializes `PricingPolicy` actors and critics
3. Runs 50 episodes with:
   - 288 steps per episode (24 hours)
   - Trajectory collection for policy gradient
   - Return computation with discount factor γ=0.99
   - Actor/critic updates with advantage estimation

**Output:**
```
[station_0] t_step=000 reg=+0.120 headroom_up=0.850 headroom_down=0.800
[station_1] t_step=000 reg=+0.120 headroom_up=0.850 headroom_down=0.800
Episode  10 | Total reward:   245.67 | Per-station: {'station_0': 118.23, 'station_1': 127.44}
Episode  20 | Total reward:   312.45 | Per-station: {'station_0': 155.20, 'station_1': 157.25}
...
Training completed.
```

#### 3. Training with Ray RLlib (Optional)

Requires Ray RLlib installation. Uses distributed PPO training:

```bash
pip install 'ray[rllib]'
python -m case_studies.power.ev_public_charging_case.train_rllib --rllib
```

**RLlib configuration:**
- Algorithm: PPO (Proximal Policy Optimization)
- Learning rate: 1e-4
- Gamma: 0.99
- Train batch size: 4000
- Entropy coefficient: 0.01
- 2 environment runners with distributed experience collection

#### 4. Event-Driven Deployment

After training, deploy with asynchronous event-driven scheduling:

```bash
python -m case_studies.power.ev_public_charging_case.train_rllib --event-driven
```

**What happens:**
1. Trains policies in synchronous CTDE mode
2. Attaches trained policies to coordinators
3. Configures `ScheduleConfig` with jitter for realistic async timing:
   - System level: 300s tick interval
   - Coordinator level: 300s tick + obs/act delays
   - Field level: Faster clock with smaller delays
4. Runs event-driven simulation via `env.run_event_driven()`

## Training Details

### CTDE Training Pipeline

Each episode follows:

```
1. env.reset() → obs, info
   │
2. for step in episode:
   │
   ├─ Policy.forward(obs) → action for each coordinator
   │
   ├─ env.step(actions)
   │  ├─ system_agent.execute(actions, proxy)
   │  │  └─ layer_actions → act (hierarchical action flow)
   │  │
   │  ├─ run_simulation(env_state)
   │  │  ├─ EV arrivals (Poisson)
   │  │  ├─ Charging dynamics (power flow)
   │  │  └─ Departures (SOC target reached)
   │  │
   │  └─ compute_observations/rewards
   │
   ├─ Accumulate: obs_t, action_t, reward_t
   │
   └─ Continue until episode_length reached
   
3. Post-episode updates:
   ├─ Compute returns: G_t = r_t + γ*G_{t+1}
   ├─ Baseline: b_t = Critic(obs_t)
   ├─ Advantage: A_t = G_t - b_t
   └─ Update actor & critic networks
```

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_stations` | 2 | Number of independent stations |
| `num_chargers` | 5 | Chargers per station |
| `arrival_rate` | 10.0 | EVs/hour Poisson rate |
| `dt` | 300.0 | Timestep in seconds (5 min) |
| `episode_length` | 86400.0 | Episode duration in seconds (24 hours) |
| `gamma` | 0.99 | Discount factor for returns |
| `lr` | 0.01 | Learning rate for policy gradient |
| `hidden_dim` | 32 | Actor/critic network width |
| `num_episodes` | 50 | Training episodes |

### Reward Structure

Station coordinator reward = **sum of subordinate slot rewards**

Slot reward components:
- **Revenue:** kWh charged × price
- **Efficiency:** Penalty for idle slots
- **Service:** Reward for successful charging completion

(Exact composition determined by reward computation in agent)

## File Structure

```
ev_public_charging_case/
├── agents/
│   ├── __init__.py              # Exports ChargingSlot, StationCoordinator
│   ├── charging_slot.py         # Field agent for single charger
│   └── station_coordinator.py   # Coordinator agent managing slots
│
├── envs/
│   ├── __init__.py
│   ├── charging_env.py          # Main HERON HeronEnv implementation
│   ├── common.py                # SlotState, EnvState dataclasses
│   ├── market_scenario.py       # Market price simulation (LMP)
│   └── regulation_scenario.py   # Frequency regulation signals
│
├── features/
│   ├── __init__.py
│   ├── charger_feature.py       # Physical charger state
│   ├── ev_slot_feature.py       # EV battery/occupancy state
│   ├── station_feature.py       # Station-level metrics
│   ├── market_feature.py        # Market signals (LMP, forecast)
│   └── regulation_feature.py    # Regulation metrics (signal, headroom)
│
├── policies/
│   ├── __init__.py
│   └── pricing_policy.py        # Actor-critic neural policy
│
├── utils/
│   ├── __init__.py              # Normalization/utility functions
│   └── ...
│
├── train_rllib.py               # Training scripts (simple + RLlib + event-driven)
├── run_single_station_rollout.py # Quick test script
├── run_event_driven.py          # Event-driven deployment example
└── README.md                    # This file
```

## Data Flow

### Synchronous CTDE Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING EPISODE LOOP                               │
└─────────────────────────────────────────────────────────────────────────────┘

Step t:
────────────────────────────────────────────────────────────────────────────────

  1. OBSERVATION COLLECTION
  ──────────────────────────
  ChargingEnv.compute_observations()
         ↓
  For each agent (coordinator & slots):
     ├─ ChargerFeature → [power_norm, status]
     ├─ EVSlotFeature → [occupied, soc, arrival_time, price_sensitivity]
     ├─ ChargingStationFeature → [open_norm, price_norm]
     ├─ MarketFeature → [lmp_norm, time_of_day, forecast]
     └─ RegulationFeature → [signal, headroom_up, headroom_down]
         ↓
  observation_dict = {
      "station_0": np.array([...8D...]),
      "station_0_slot_0": np.array([...]),
      "station_0_slot_1": np.array([...]),
      ...
  }


  2. POLICY INFERENCE
  ──────────────────
  For each coordinator agent:
     ├─ obs = observation_dict["station_i"]
     ├─ PricingPolicy.forward(obs)
     │   ├─ h = ReLU(obs @ W1 + b1)      [32D hidden]
     │   └─ action = sigmoid(h @ W2 + b2) * 0.8  [1D ∈ [0.0, 0.8]]
     └─ action_dict["station_i"] = price_action


  3. ACTION TRANSMISSION (Vertical Protocol)
  ──────────────────────────────────────────
  system_agent.execute(action_dict, proxy)
     ├─ SystemAgent.set_action(action_dict)
     └─ BroadcastActionProtocol.act()
         ├─ For each coordinator:
         │   ├─ price = action_dict["station_i"].c[0]  [scalar price]
         │   └─ Broadcast(price) → all subordinate slots
         │
         └─ For each slot:
             ├─ Receive price from coordinator
             ├─ ChargingSlot.set_action(price)
             └─ slot.action.c[0] = price


  4. ENVIRONMENT SIMULATION
  ─────────────────────────
  ChargingEnv.run_simulation(env_state)
     │
     ├─ EV ARRIVALS
     │  ├─ Poisson process: λ = arrival_rate
     │  ├─ For each new arrival:
     │  │   ├─ Assign to available slot (random)
     │  │   ├─ Initialize: occupied=1, soc=0.1, soc_target=0.8
     │  │   └─ Sample: price_sensitivity ~ U[0.0, 1.0]
     │  └─ new_arrivals_count = |arrivals this step|
     │
     ├─ CHARGING DYNAMICS
     │  ├─ For each occupied slot i:
     │  │   ├─ Read: price_i, soc_i, p_max_i, price_sensitivity_i
     │  │   ├─ Compute willingness:
     │  │   │   willingness = 1.0 - (price_i * price_sensitivity_i / max_price)
     │  │   │   (Higher prices → lower willingness)
     │  │   │
     │  │   ├─ Compute power:
     │  │   │   p_kw = p_max_i * max(0, willingness)
     │  │   │
     │  │   ├─ Update SOC:
     │  │   │   Δ_soc = (p_kw * dt) / battery_capacity
     │  │   │   soc_new = soc_i + Δ_soc
     │  │   │
     │  │   ├─ Accumulate revenue:
     │  │   │   revenue += p_kw * price_i * (dt / 3600.0)
     │  │   │
     │  │   └─ Accumulate power for station:
     │  │       station_power[station_i] += p_kw
     │  │
     │  └─ Update all slot states in env_state
     │
     ├─ EV DEPARTURES
     │  ├─ For each occupied slot i:
     │  │   └─ If soc_new >= soc_target OR wait_time > max_wait:
     │  │       ├─ Mark slot as unoccupied: occupied = 0
     │  │       ├─ Reset SOC: soc = 0.0
     │  │       └─ Add departure event (optional logging)
     │
     └─ Update global state:
        ├─ self._time_s += dt
        ├─ self.scenario.step()  → updates LMP
        └─ self.reg_scenario.step()  → updates regulation signal


  5. STATE UPDATE (Agent Internals)
  ─────────────────────────────────
  For each agent:
     ├─ ChargingSlot agents:
     │   └─ state.update_feature(
     │       "ChargerFeature", p_kw=...,
     │       "EVSlotFeature", occupied=..., soc=..., ...)
     │
     └─ StationCoordinator:
         └─ state.update_feature(
             "ChargingStationFeature", open_chargers=..., price=...,
             "MarketFeature", lmp=...,
             "RegulationFeature", signal=..., ...)


  6. REWARD COMPUTATION
  ──────────────────────
  ChargingEnv.compute_rewards(proxy)
     │
     ├─ For each ChargingSlot agent:
     │   ├─ Reward components:
     │   │   ├─ r_revenue = revenue_this_step
     │   │   ├─ r_efficiency = penalty if p_kw=0 while occupied
     │   │   └─ r_service = bonus if soc_new >= soc_target
     │   │
     │   └─ slot_reward = w_rev*r_revenue + w_eff*r_efficiency + w_svc*r_service
     │
     └─ For each StationCoordinator:
         └─ coord_reward = sum(slot_reward for all subordinates)
             [Implements hierarchical reward aggregation]
             
    reward_dict = {
        "station_0": coord_reward_0,
        "station_0_slot_0": slot_reward_00,
        "station_0_slot_1": slot_reward_01,
        ...
    }


  7. TRAJECTORY RECORDING
  ──────────────────────
  Accumulate in episode buffer:
     ├─ transitions.append({
     │    'obs': observation_dict,
     │    'action': action_dict,
     │    'reward': reward_dict,
     │    'next_obs': next_observation_dict,
     │    'done': truncated["__all__"]
     │  })
     │
     └─ Continue to next step


────────────────────────────────────────────────────────────────────────────────

POST-EPISODE UPDATE:
────────────────────────────────────────────────────────────────────────────────

  8. RETURN COMPUTATION (Backward Pass)
  ────────────────────────────────────
  For each coordinator policy:
     ├─ G_T = 0  (terminal value)
     │
     ├─ for t in T-1 down to 0:
     │   ├─ G_t = r_t + γ * G_{t+1}
     │   └─ returns[t] = G_t
     │
     └─ returns = [G_0, G_1, ..., G_T]


  9. ADVANTAGE ESTIMATION
  ──────────────────────
  For each step t:
     ├─ value_t = Critic(obs_t)
     ├─ advantage_t = G_t - value_t
     │   [Returns - baseline: variance reduction while maintaining unbiased gradient]
     │
     └─ advantages = [A_0, A_1, ..., A_T]


  10. POLICY GRADIENT UPDATE
  ──────────────────────────
  For each coordinator:
     ├─ Actor update:
     │   ├─ For each transition (obs, action, advantage):
     │   │   ├─ pred_action = Actor.forward(obs)
     │   │   ├─ error = pred_action - action  [Supervised regression toward taken action]
     │   │   ├─ weighted_error = error * advantage
     │   │   └─ Backprop: ∇ Actor.params ∝ weighted_error
     │   │
     │   └─ Actor.update(X, action, advantage, lr=0.01)
     │       [Policy gradient: nudge actor toward high-advantage actions]
     │
     └─ Critic update:
         ├─ For each transition (obs, return):
         │   ├─ pred_value = Critic.forward(obs)
         │   ├─ value_error = pred_value - return
         │   └─ Backprop: ∇ Critic.params ∝ value_error
         │
         └─ Critic.update(X, return, lr=0.01)
             [Baseline fitting: predict returns better]


  11. EXPLORATION NOISE DECAY (Optional)
  ──────────────────────────────────────
  If using epsilon-greedy or Gaussian noise:
     ├─ epsilon = epsilon_init * decay_factor^episode
     └─ Use in policy: action += epsilon * N(0,1)


────────────────────────────────────────────────────────────────────────────────

Return to Step 1 for next episode (reset environment)

```

### Event-Driven Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EVENT-DRIVEN DEPLOYMENT                                │
│          (After training: attach policies, configure tick configs)           │
└─────────────────────────────────────────────────────────────────────────────┘

INITIALIZATION:
───────────────
1. env.run_event_driven(num_steps=1000)
2. Scheduler registers all agents with ScheduleConfig
3. Initialize event queue with first tick events


RUNTIME LOOP (Scheduler-driven):
────────────────────────────────

While scheduler.step():
    │
    ├─ next_event = scheduler.pop_earliest_event()
    │  └─ Event = (time_ts, agent_id, event_type)  where event_type ∈ {OBS, ACT, MSG, REWARD, ...}
    │
    ├─ Event dispatch:
    │  │
    │  ├─ OBS events (Observation):
    │  │  ├─ agent.compute_observations()
    │  │  ├─ cached_obs[agent_id] = obs
    │  │  └─ Schedule next ACT event at time_ts + act_delay
    │  │
    │  ├─ ACT events (Action):
    │  │  ├─ if isinstance(agent, Coordinator):
    │  │  │   ├─ obs = cached_obs[agent_id]
    │  │  │   ├─ action = agent.policy.forward(obs)  [Trained policy!]
    │  │  │   ├─ agent.set_action(action)
    │  │  │   └─ Broadcast to subordinates via protocol
    │  │  │
    │  │  └─ if isinstance(agent, FieldAgent):
    │  │      ├─ Receive broadcasted action from coordinator (via MSG)
    │  │      ├─ agent.set_action(action)
    │  │      └─ No further action needed (passive)
    │  │
    │  ├─ MSG events (Messaging):
    │  │  ├─ protocol.receive_message(sender, message)
    │  │  ├─ Update agent's action/state based on message
    │  │  └─ Schedule next event (ACT, OBS, or REWARD)
    │  │
    │  └─ REWARD events (Reward computation):
    │     ├─ agent.compute_rewards(proxy)
    │     ├─ Store reward (for logging/analysis)
    │     └─ Schedule next OBS event
    │
    ├─ Simulation step (when all agents acted):
    │  ├─ env.run_simulation(env_state)  [Compute physics]
    │  └─ Update agent states from simulation
    │
    └─ Continue while event_queue not empty and num_steps < limit


DATA FLOW EXAMPLE (Two coordinators, 2 steps):
───────────────────────────────────────────────

t=0s: SystemAgent OBS
   ├─ → SystemAgent ACT (t=2s)
   │
   t=1s: Station0 OBS
   │  → Station0 ACT (t=3s)
   │
   t=1s: Station1 OBS
      → Station1 ACT (t=3s)

t=2s: SystemAgent ACT
   ├─ Broadcast actions to Stations
   │  → Station0 MSG (t=3.5s)
   │  → Station1 MSG (t=3.5s)

t=3s: Station0 ACT
   ├─ Send actions to subordinate slots
   │  → Slot0 MSG (t=3.5s)
   │  → Slot1 MSG (t=3.5s)
   │
t=3s: Station1 ACT
      → Slot2 MSG (t=3.5s)
      → Slot3 MSG (t=3.5s)

t=3.5s: Slot0 MSG (receive price)
   → Slot0 REWARD (t=7.5s)

... (similar for other slots)

t=7.5s: All field agents done
   → env.run_simulation() [Compute physics at this moment]
   → Slot rewards aggregated to coordinator rewards
   → Station0 REWARD, Station1 REWARD

t=8s: Station0 OBS (next tick cycle)
   → Station0 ACT (t=10s)
   ...

```

### Data Structure Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT STATE                              │
│  (ChargingEnv._slot_to_station, scenario, reg_scenario)          │
└──────────────────────────────────────────────────────────────────┘
                             ↑↓
        ╔═════════════════════════════════════════════════╗
        ║          env_state: EnvState                     ║
        ║  ┌──────────────────────────────────────────┐   ║
        ║  │ slot_states: Dict[slot_id, SlotState]    │   ║
        ║  │  ├─ p_kw: float          (power output)  │   ║
        ║  │  ├─ occupied: int        (1 if EV here)  │   ║
        ║  │  ├─ soc: float           (0.0-1.0)       │   ║
        ║  │  ├─ soc_target: float    (target charge) │   ║
        ║  │  ├─ price_sensitivity: float (0-1)       │   ║
        ║  │  └─ revenue: float       (accumulated)   │   ║
        ║  │                                          │   ║
        ║  │ station_prices: Dict[station_id, float]  │   ║
        ║  │  └─ price ∈ [0.0, 0.8] $/kWh             │   ║
        ║  │                                          │   ║
        ║  │ lmp: float               (market price)  │   ║
        ║  │ time_s: float            (simulation time)│  ║
        ║  │ dt: float                (timestep)      │   ║
        ║  │ reg_signal: float        (frequency dev) │   ║
        ║  │ new_arrivals: int        (this step)     │   ║
        ║  └──────────────────────────────────────────┘   ║
        ╚═════════════════════════════════════════════════╝
                             ↓
    ┌────────────────────────────────────────────────┐
    │      Run Simulation Phase (Physics)             │
    │  (env.run_simulation → updates env_state)       │
    └────────────────────────────────────────────────┘
                             ↓
    ┌────────────────────────────────────────────────┐
    │      Observation Computation Phase              │
    │   (For each agent, extract features)            │
    └────────────────────────────────────────────────┘
       ↓                    ↓                    ↓
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Coordinator  │  │ Coordinator  │  │ FieldAgent   │
    │ ("station_0")│  │ ("station_1")│  │ ("slot_x")   │
    └──────────────┘  └──────────────┘  └──────────────┘
       ↓ obs(8D)         ↓ obs(8D)        ↓ obs(nD)
       │                 │                │
       ├─ Policy.forward()  ├─ Policy.forward()
       │                 │
    action(1D)        action(1D)
    [price ∈           [price ∈
     [0,0.8]]          [0,0.8]]
       │                 │
       └─────┬───────────┘
             ├─ Broadcast to subordinates
             │
    ┌────────────────────────────────────┐
    │  slot_0.action = price             │
    │  slot_1.action = price             │
    │  ...                               │
    └────────────────────────────────────┘
             ↓ (apply_action → simulation)
    ┌────────────────────────────────────────────────┐
    │     Reward Computation Phase                    │
    │  (compute_rewards for each agent)               │
    └────────────────────────────────────────────────┘
       ↓                    ↓                    ↓
    reward(scalar)   reward(scalar)    reward(scalar)
   (coordinator)    (coordinator)     (field agent)
       │                 │                │
       └─────┬───────────┴─────┬──────────┘
             └──────────────────┘
                  reward_dict
                  aggregated
```

## Key Concepts

### State, Action, Observation

| Level | Observation (Input) | Action (Output) | Features |
|-------|-------------------|-----------------|----------|
| **Coordinator** | 8D: station state + market + regulation | 1D: price [0.0, 0.8] $/kWh | ChargingStationFeature, MarketFeature, RegulationFeature |
| **Slot** | Broadcast price + EV state | 1D: follows price signal | ChargerFeature, EVSlotFeature |

### Reward Signals

Rewards are computed after simulation each step:
- **Agent-centric:** Each agent receives reward for its actions
- **Hierarchical aggregation:** Coordinator reward sums subordinate rewards
- **Signal components:** Revenue, efficiency, fairness, constraint compliance

### Observation Features (8D)

1. **[0]** Open chargers (normalized): `open_chargers / max_chargers`
2. **[1]** Current price (normalized): `(price - 0.0) / (0.8 - 0.0)`
3. **[2]** Current LMP (normalized)
4. **[3]** Time-of-day factor
5. **[4]** Next period LMP forecast
6. **[5]** Regulation signal (frequency deviation)
7. **[6]** Regulation headroom up
8. **[7]** Regulation headroom down

### Price-Demand Relationship

EVs have **price sensitivity**:
- High price → lower willingness to charge → lower load
- Low price → higher willingness to charge → higher load
- Sensitivity varies by EV type (parameter range: 0.0-1.0)

Policy learns to balance:
- **High prices** → higher revenue per kWh but lower volume
- **Low prices** → higher volume but lower margin

## Advanced Usage

### Custom Environment Configuration

```python
env = ChargingEnv(
    coordinator_agents=[coordinator1, coordinator2],
    arrival_rate=15.0,      # More arrivals
    dt=600.0,               # 10-minute steps
    episode_length=172800.0,  # 48 hours
    reg_freq=2.0,           # Regulation signal update frequency
    reg_alpha=0.15,         # Regulation amplitude
    seed=123,
)
```

### Using Trained Policies

```python
from case_studies.power.ev_public_charging_case.policies import PricingPolicy

# Load or create policy
policy = PricingPolicy(obs_dim=8, action_dim=1, hidden_dim=32, seed=42)

# Forward pass
obs = env.reset()[0]
action = policy.forward(obs["station_0"])
print(f"Recommended price: ${action.c[0]:.2f}/kWh")
```

### Custom Reward Function

Override `compute_rewards()` in your agent:

```python
class CustomCoordinator(StationCoordinator):
    def compute_rewards(self, proxy):
        rewards = {}
        for subordinate in self.subordinates.values():
            rewards[subordinate.agent_id] = custom_reward_fn(subordinate.state)
        rewards[self.agent_id] = sum(rewards.values()) * weight_factor
        return rewards
```

## Logging & Analysis

Training scripts produce detailed logs:

```
2024-03-03 10:23:45 [INFO] Station agents: ['station_0', 'station_1']
2024-03-03 10:23:46 [INFO] [station_0] t_step=000 reg=+0.120 headroom_up=0.850 headroom_down=0.800
2024-03-03 10:23:50 [INFO] Episode  10 | Total reward:   245.67 | Per-station: {...}
```

Access detailed logs via:
```python
logging.basicConfig(level=logging.DEBUG)
```

For event-driven analysis, use HERON's `EpisodeAnalyzer`:

```python
from heron.scheduling.analysis import EpisodeAnalyzer

analyzer = EpisodeAnalyzer(env.scheduler)
analyzer.print_summary()
analyzer.plot_timeline()  # Requires matplotlib
```

## Troubleshooting

### ImportError: No module named 'heron'

```bash
pip install -e /Users/yulinzeng/PycharmProjects/PowerGym
```

### Ray RLlib errors during training

Install Ray with RLlib support:
```bash
pip install 'ray[rllib]>=2.10.0'
```

### Low rewards during initial training

This is normal! PricingPolicy starts with random exploration. Rewards should increase over 50 episodes. Try:
- Increase `num_episodes` to 100+
- Lower learning rate to 0.001 for stable updates
- Check feature normalization in `features/*.py`

### Event-driven deployment shows no progress

Verify:
1. Policies are attached: `coordinator.policy is not None`
2. ScheduleConfig is configured: `agent.schedule_config is not None`
3. Scheduler has events: `len(env.scheduler.event_queue) > 0`

## References

- **HERON Documentation:** Framework for hierarchical multi-agent RL
- **PowerGym Case Studies:** Other examples in `case_studies/power/`
- **RLlib Docs:** Ray distributed training framework

## Contributing

To extend this case study:

1. **New agent types:** Subclass `FieldAgent` or `CoordinatorAgent`
2. **New features:** Implement `Feature` interface
3. **New policies:** Subclass `Policy` with custom forward/update
4. **New scenarios:** Create scenario class with market/regulation dynamics

See HERON's `heron/core/` and `heron/agents/` for API details.

## License

See LICENSE.txt in the PowerGym root directory.

---

**Version:** 1.0  
**Last Updated:** March 3, 2024  
**Maintainer:** PowerGym Team


