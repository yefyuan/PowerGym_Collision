# Experiment Briefing: Measuring the Step-to-Deployment Gap

---

## What We're Proving

The paper's central claim: **MAPPO policies scoring near-optimal under step-based evaluation degrade by 8–15% under realistic SCADA timing, and we can decompose which deployment parameters drive the degradation.**

This requires three things from the experiments:
1. A **baseline** (step-based score)
2. A **deployment score** (event-driven with realistic timing)
3. A **per-parameter decomposition** showing which factor contributes what

---

## The Event Model — Why Policies Break

In training, `env.step()` is atomic — observe, decide, act, physics update all happen at the same instant for all agents. At deployment, these are **separate events of different types** that interleave on a timeline.

Heron's event taxonomy classifies CPS deployment events into 5 classes by what determines *when* they fire:

| Class | What determines timing | Deployment examples | Heron event types |
|---|---|---|---|
| **1 — Periodic** | Timer / calendar | Agent sensor poll, SCADA scan cycle, coordinator dispatch | `AGENT_TICK` |
| **2 — Consequence** | Delayed result of a prior event | Action takes effect, obs delivered, message arrives | `ACTION_EFFECT`, `OBSERVATION_READY`, `MESSAGE_DELIVERY` |
| **3 — Condition** | State threshold crossed | Voltage violation alarm, relay trip | `CONDITION_TRIGGER` *(planned — see Gap A)* |
| **4 — Exogenous** | External to the system | Line fault, load spike, weather change | `ENV_UPDATE` *(planned — see Gap B)* |
| **5 — Physics** | Solver schedule | Power flow solve, traffic sim step | `SIMULATION` |

Step-based training collapses all 5 classes into one synchronous tick. Event-driven evaluation separates them.

**How they chain in a real scenario:**
```
[4] Line fault at t=12.3              (exogenous — topology changes)
  → [5] Physics re-solves at t=12.3    (new power flow with fault)
  → [3] V < 0.95 detected at t=12.3   (condition trigger — alarm)
  → [1] Agent wakes at t=13.0         (next periodic tick — 0.7s late)
  → [2] Action effect at t=13.5       (consequence — 0.5s actuator delay)
```
Total: 1.2 seconds from fault to corrective action. In step-based training this is zero.

**Classes 1, 2, 5 are implemented and ablatable now.** Classes 3, 4 are planned — see [implementation TODO](impl_todo_2026_03_20.md).

---

## Three Failure Modes to Measure

Each maps to event classes and a `ScheduleConfig` parameter:

| Failure mode | Event classes involved | Parameter | What goes wrong |
|---|---|---|---|
| **Physics drift** | 1→2→5 (tick → delayed effect → physics evolved) | `act_delay` | Agent observes s₁, but action lands on s₄ — decision was for a world that no longer exists |
| **Late coordination** | 1→2 (independent ticks → message latency) | `msg_delay` | Coordinator's directive arrives after subordinate already acted on its own cycle |
| **Scheduling jitter** | 1 (periodic with jitter) | `jitter_ratio` | Agents drift out of phase — relative timing becomes unpredictable |

---

## The Experiments

### Step 0: Train the baseline

Train with MAPPO (or QMIX) in step-based mode. Verify convergence. Record the **step-based evaluation score**.

### Step 1: Single-parameter sweeps (Classes 1 & 2)

Hold all other parameters at zero. Sweep one at a time → **per-parameter degradation curves**.

| ID | Parameter | Event class | Values | What it isolates |
|---|---|---|---|---|
| **E1** | `act_delay` | 2 — Consequence | 0, 0.1, 0.25, 0.5, 1.0 s | Physics drift: how much does actuator delay cost? |
| **E2** | `msg_delay` | 2 — Consequence | 0, 0.05, 0.1, 0.5, 1.0 s | Late coordination: when does latency break hierarchy? |
| **E3** | `jitter_ratio` | 1 — Periodic | 0, 0.05, 0.1, 0.2, 0.3 | Scheduling drift: how much jitter before desynchronization? |
| **E4** | `tick_interval` | 1 — Periodic | 1:1, 1:10, 1:60, 1:300 | Tick heterogeneity: does rate mismatch matter independently? |

**What to look for**: E1 and E3 are expected to be dominant. E2 should show a threshold — fine below some latency, then coordination collapses. E4 tests whether the Class 1 rate ratio matters independently of Class 2 delays.

### Step 2: Interaction experiments (Class 1 × Class 2)

Do parameters from different event classes compound or substitute? This is what makes the decomposition story convincing.

| ID | Design | Classes crossed | Question |
|---|---|---|---|
| **E5** | 3×3 grid: `act_delay` ∈ {0, 0.25, 0.5} × `jitter_ratio` ∈ {0, 0.1, 0.2} | 2 × 1 | Is consequence delay + periodic jitter worse than the sum? |
| **E6** | 3×3 grid: `msg_delay` ∈ {0, 0.1, 0.5} × `tick_interval` ratio ∈ {1:1, 1:60, 1:300} | 2 × 1 | Does coordination break faster with heterogeneous rates? |

**What to look for**: Positive interaction term (gap > sum of individual gaps) = compounding. This is the strongest evidence that step-based evaluation is misleading — each parameter alone looks manageable, but together they're worse.

### Step 3: Realistic deployment profile (all classes together)

Set **all** parameters to domain-realistic values simultaneously. This is the headline number.

**Power systems (SCADA timing)**:

| Agent tier | `tick_interval` (Class 1) | `act_delay` (Class 2) | `msg_delay` (Class 2) | `jitter_ratio` (Class 1) |
|---|---|---|---|---|
| FieldAgent | 1.0 s | 0.3 s | 0.05 s | 0.10 |
| CoordinatorAgent | 60.0 s | 0.0 s | 0.10 s | 0.05 |
| SystemAgent | 300.0 s | 0.0 s | 0.20 s | 0.02 |

The gap between this and the step-based baseline = the paper's main result.

---

## How to Run

```python
# Step-based (training + baseline evaluation)
obs, _ = env.reset()
for step in range(num_steps):
    actions = {aid: policy(obs[aid]) for aid in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)

# Event-driven (deployment evaluation) — same agents, same policies
obs, _ = env.reset()
env.set_agent_policies({aid: policy for aid in env.agents})
result = env.run_event_driven(t_end=30.0)
```

No code changes between modes. Swap `ScheduleConfig` values between experiments.

---

## What to Report

| Metric | Source |
|---|---|
| Step-based score | Baseline (Step 0) |
| Event-driven score | Per parameter value |
| Gap (%) | (step − event) / step × 100 |
| Dominant parameter | Which E1–E4 shows largest gap |
| Interaction term | E5/E6: gap(both) − gap(A) − gap(B) + gap(neither) |
| Realistic gap | Step 3: the headline number |

**The paper needs**:
- **Degradation curves** per parameter (E1–E4) — line plots, one per event class
- **Heatmap** for at least one interaction (E5 or E6) — shows compounding across classes
- **Headline gap** under realistic SCADA timing (Step 3)
- **Decomposition bar chart** — fraction of realistic gap attributable to each parameter/class

---

## Priority

If time is limited, run in this order:

| Priority | Experiment | Why |
|---|---|---|
| 1 | **Step 3** (realistic profile) | The headline result — needed for abstract and intro claims |
| 2 | **E1** (`act_delay` sweep) | Expected dominant factor — Class 2 consequence delay |
| 3 | **E3** (`jitter_ratio` sweep) | Expected second factor — Class 1 periodic drift |
| 4 | **E5** (`act_delay` × `jitter_ratio`) | Demonstrates Class 1 × Class 2 compounding |
| 5 | **E2** (`msg_delay` sweep) | Coordination breakdown threshold — Class 2 |
| 6 | E4, E6 | Supporting evidence, lower priority |
