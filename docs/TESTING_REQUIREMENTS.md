# HERON Testing Requirements

This document defines the complete set of files to run and their passing criteria to verify that changes to HERON components do not break existing functionality.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Unit Tests (pytest)](#1-unit-tests-pytest)
3. [Integration Tests (pytest)](#2-integration-tests-pytest)
4. [Case Study Tests (pytest)](#3-case-study-tests-pytest)
5. [Case Study Run Scripts](#4-case-study-run-scripts)
6. [Framework Examples (Levels 1-7)](#5-framework-examples-levels-1-7)
7. [Jupyter Notebooks - Framework](#6-jupyter-notebooks---framework)
8. [Jupyter Notebooks - Power Grid Tutorials](#7-jupyter-notebooks---power-grid-tutorials)
9. [Running Everything](#running-everything)

---

## Quick Reference

| Category | Count | Command |
|----------|-------|---------|
| Unit tests | 4 tests in 1 file | `pytest tests/test_env_builder.py -v` |
| Integration tests | 5 files (manual + pytest) | See [Section 2](#2-integration-tests-pytest) |
| Case study tests | 1 file | `pytest case_studies/power/tests/test_hierarchical_env.py -v` |
| Case study scripts | 3 scripts | `python case_studies/power/ev_public_charging_case/*.py` |
| Framework examples | 20 scripts | `python examples/<N>.*/<script>.py` |
| Framework notebooks | 2 notebooks | `examples/notebooks/` |
| Power grid notebooks | 6 notebooks | `case_studies/power/tutorials/` |
| **Total** | **37 runnable items** | |

---

## 1. Unit Tests (pytest)

### `tests/test_env_builder.py`

**Run command:**
```bash
pytest tests/test_env_builder.py -v
```

| Test | What It Validates | Passing Criteria |
|------|-------------------|------------------|
| `test_builder_callable_returns_env` | `EnvBuilder()` returns a callable that produces a valid `HeronEnv` | `env is not None`; `obs` is a `dict` after `reset()` |
| `test_builder_callable_with_config_arg` | Callable accepts an optional config dict (RLlib compatibility) | `env is not None`; no errors when passed `{"some_key": 123}` |
| `test_builder_callable_produces_independent_envs` | Each call creates independent environment instances | `env1 is not env2`; both reset independently without side effects |
| `test_builder_with_custom_coordinator_no_protocol_conflict` | Builder does not pass `protocol=` when `None`, avoiding `TypeError` on custom coordinators | `env is not None`; resets without `TypeError` |

**Pass = All 4 tests green.**

---

## 2. Integration Tests (pytest)

### 2.1 `tests/integration/test_action_passing.py`

**Run command:**
```bash
python tests/integration/test_action_passing.py
```

**What it tests:** CTDE training (50 episodes, 30 steps) + event-driven execution (100s). A coordinator learns to control 2 devices, minimizing power deviation from zero via `VerticalProtocol` action decomposition.

| Criteria | Assertion |
|----------|-----------|
| Training completes | 50 episodes run without error |
| Return improves | `mean(returns[-10:]) > mean(returns[:10])` (or close) |
| Power converges toward 0 | `abs(mean(avg_powers[-10:])) < abs(mean(avg_powers[:10]))` |
| Event-driven executes | Coordinator ticks ~25 times over 100s (tick_interval=4s) |
| Event activity | >0 observations, state requests, and action results logged |

---

### 2.2 `tests/integration/test_e2e.py`

**Run command:**
```bash
python tests/integration/test_e2e.py
```

**What it tests:** End-to-end CTDE with per-agent (decentralized) policies. 2 battery agents learn to maximize SOC using local-only observation mode.

| Criteria | Assertion |
|----------|-----------|
| Training completes | 100 episodes (50 steps each) run without error |
| SOC improves | `mean(avg_socs[-10:]) > mean(avg_socs[:10])` |
| Event-driven executes | Asynchronous agent ticks with jitter complete |
| Policies deploy | Trained policies attach and run in event-driven mode |

---

### 2.3 `tests/integration/test_maddpg_action_passing.py`

**Run command:**
```bash
python tests/integration/test_maddpg_action_passing.py
```

**What it tests:** MADDPG (centralized critic, decentralized actors) training on the 2-device action-passing environment with continuous actions.

| Criteria | Assertion |
|----------|-----------|
| Training completes | 500 episodes (30 steps each) |
| Reward improves | `mean(rewards[-50:]) > mean(rewards[:50])` or improvement >= -0.5 |
| Env sanity | Agent IDs = `["device_1", "device_2"]`; obs shape = (2,); action shape = (1,) |
| Replay buffer | >=500 transitions collected before training begins |

---

### 2.4 `tests/integration/test_qmix_action_passing.py`

**Run command:**
```bash
python tests/integration/test_qmix_action_passing.py
```

**What it tests:** QMIX value decomposition training with 11-action discrete spaces on the 2-device environment.

| Criteria | Assertion |
|----------|-----------|
| Training completes | 600 episodes (30 steps each) |
| Reward improves | `mean(rewards[-50:]) > mean(rewards[:50])` or improvement >= -1.0 |
| Action space | Discrete with 11 actions (indices 0-10) |
| Target network sync | Hard updates every 200 training steps |
| Epsilon schedule | Decays from 1.0 to 0.02 over 4000 steps |

---

### 2.5 `tests/integration/test_rllib_action_passing.py`

**Run command:**
```bash
python tests/integration/test_rllib_action_passing.py
```

**What it tests:** `RLlibBasedHeronEnv` adapter with PPO (both MAPPO shared-policy and IPPO independent-policy configurations).

| Criteria | Assertion |
|----------|-----------|
| Sanity check passes | obs/reward/terminated/truncated dict keys match agent IDs; correct shapes |
| MAPPO trains | 5 iterations of `algo.train()` complete; `mean_reward` is not NaN |
| IPPO trains | 5 iterations of `algo.train()` complete; `mean_reward` is not NaN |
| Agent IDs | `["device_1", "device_2"]` present in all dicts |

---

## 3. Case Study Tests (pytest)

### `case_studies/power/tests/test_hierarchical_env.py`

**Run command:**
```bash
python case_studies/power/tests/test_hierarchical_env.py
```

**What it tests:** Full hierarchical microgrid system (SystemAgent -> 3 PowerGridAgents -> multiple Generator/ESS devices) with PandaPower network simulation, CTDE training (30 episodes, 24 steps), and event-driven evaluation (300s).

| Criteria | Assertion |
|----------|-----------|
| Training completed | `len(returns) == 30` |
| Learning occurred | `mean(returns[-5:]) > mean(returns[:5]) - 1.0` |
| All microgrids active | `len(system_agent.subordinates) == 3` |
| Coordinator policies trained | `len(coordinator_policies) == 3` |
| Event-driven ran | observations > 0, actions > 0 |
| Power flow convergence | >90% of power flow computations converge |

---

## 4. Case Study Run Scripts

### 4.1 `case_studies/power/ev_public_charging_case/run_single_station_rollout.py`

**Run command:**
```bash
python case_studies/power/ev_public_charging_case/run_single_station_rollout.py
```

**What it does:** Single-station rollout with random policy. 1 station, 5 charger slots, 288 steps (1 day at 300s intervals).

| Criteria | How to Verify |
|----------|---------------|
| Runs to completion | Script finishes without error |
| Observations correct | `Agents in obs: ['station_0']` printed |
| Reward is finite | Total reward over 288 steps is a finite number (not NaN/Inf) |
| Step rewards printed | Rewards printed every 50 steps show plausible values |

---

### 4.2 `case_studies/power/ev_public_charging_case/train_rllib.py`

**Run command:**
```bash
python case_studies/power/ev_public_charging_case/train_rllib.py
```

**What it does:** CTDE REINFORCE training on multi-station EV charging environment (50 episodes default).

| Criteria | How to Verify |
|----------|---------------|
| Training completes | 50 episodes logged without error |
| Per-station rewards logged | Each episode prints per-station reward breakdown |
| Rewards are finite | No NaN/Inf in training output |
| Optional: `--rllib` mode | `python train_rllib.py --rllib` runs RLlib PPO training |
| Optional: `--event-driven` mode | Deploys trained policies in event-driven mode |

---

### 4.3 `case_studies/power/ev_public_charging_case/run_event_driven.py`

**Run command:**
```bash
python case_studies/power/ev_public_charging_case/run_event_driven.py
```

**What it does:** Trains CTDE policies, then deploys in event-driven simulation with tick configs and jitter.

| Criteria | How to Verify |
|----------|---------------|
| Training phase completes | Last 5 episode rewards printed |
| Event-driven completes | `"Event-driven simulation complete"` printed |
| Event statistics logged | Event counts, duration, message type breakdown shown |
| Per-agent rewards present | Per-agent total reward and step counts reported |

---

## 5. Framework Examples (Levels 1-7)

All examples are standalone scripts. Each must run without error and produce meaningful output.

### Level 1: Starter (`examples/1. starter/`)

| Script | Run Command | Passing Criteria |
|--------|-------------|------------------|
| `case1.py` | `python "examples/1. starter/case1.py"` | Environment initializes, agents registered, multi-step simulation completes without error |
| `case2.py` | `python "examples/1. starter/case2.py"` | Same as case1 with extended mechanics (CBF safety filter); simulation completes |

> Note: `agents.py`, `env_physics.py`, `features.py` are module files imported by case1/case2, not standalone scripts.

### Level 2: Core Abstractions (`examples/2. core_abstractions/`)

| Script | Run Command | Passing Criteria |
|--------|-------------|------------------|
| `actions_and_spaces.py` | `python "examples/2. core_abstractions/actions_and_spaces.py"` | Continuous, discrete, and mixed action specs printed; gymnasium space interop works; scale/unscale round-trips correctly |
| `features_and_visibility.py` | `python "examples/2. core_abstractions/features_and_visibility.py"` | Feature visibility matrix printed; owner sees all own features; peer sees only public; coordinator sees upper_level; system sees system-level |
| `observations_and_state.py` | `python "examples/2. core_abstractions/observations_and_state.py"` | State composition from features works; `observed_by()` filtering returns correct subsets; serialization round-trip matches |

### Level 3: Building Environments (`examples/3. building_environments/`)

| Script | Run Command | Passing Criteria |
|--------|-------------|------------------|
| `custom_heron_env.py` | `python "examples/3. building_environments/custom_heron_env.py"` | Water tank simulation runs 20 steps; tank levels change based on pump actions; rewards are finite |
| `env_builder_patterns.py` | `python "examples/3. building_environments/env_builder_patterns.py"` | Batch registration creates correct agent count; coordinator assignment works; auto-coordinator creation works; custom system agent accepted |
| `simple_env_quickstart.py` | `python "examples/3. building_environments/simple_env_quickstart.py"` | SimpleEnv creates without custom HeronEnv subclass; room temperatures change; 10 steps complete |

### Level 4: Protocols & Coordination (`examples/4. protocols_and_coordination/`)

| Script | Run Command | Passing Criteria |
|--------|-------------|------------------|
| `custom_protocol.py` | `python "examples/4. protocols_and_coordination/custom_protocol.py"` | Weighted allocation distributes correctly (proportional to weights); threshold alerts fire for low values; composed protocol works |
| `horizontal_state_sharing.py` | `python "examples/4. protocols_and_coordination/horizontal_state_sharing.py"` | Fully-connected topology: each sensor sees all others; Ring: each sees 2 neighbors; Star: center sees all, leaves see only center |
| `vertical_action_decomposition.py` | `python "examples/4. protocols_and_coordination/vertical_action_decomposition.py"` | Vector decomposition: joint action `[a1,a2,a3]` correctly splits to per-agent slices; Broadcast: all agents receive same action |

### Level 5: Training Algorithms (`examples/5. training_algorithms/`)

| Script | Run Command | Passing Criteria |
|--------|-------------|------------------|
| `policy_and_training.py` | `python "examples/5. training_algorithms/policy_and_training.py"` | IPPO training: `mean(returns[-20:]) > mean(returns[:20])`; MAPPO training: `mean(returns[-20:]) > mean(returns[:20])`; comparison table printed |
| `rllib_integration.py` | `python "examples/5. training_algorithms/rllib_integration.py"` | `RLlibBasedHeronEnv` wraps correctly; IPPO and MAPPO configs build; 5 training iterations complete; mean_return values extracted |

### Level 6: Event-Driven Simulation (`examples/6. event_driven_simulation/`)

| Script | Run Command | Passing Criteria |
|--------|-------------|------------------|
| `dual_mode_execution.py` | `python "examples/6. event_driven_simulation/dual_mode_execution.py"` | Step-based: 10 steps complete with deterministic timing; Event-driven: simulation runs for target duration; event count breakdown printed with >0 AGENT_TICK events |
| `schedule_config_and_scheduling.py` | `python "examples/6. event_driven_simulation/schedule_config_and_scheduling.py"` | Deterministic config: exact intervals; Jittered config: samples show expected mean and std; UNIFORM and GAUSSIAN distributions produce different ranges |

### Level 7: Advanced Patterns (`examples/7. advanced_patterns/`)

| Script | Run Command | Passing Criteria |
|--------|-------------|------------------|
| `custom_env_and_visibility.py` | `python "examples/7. advanced_patterns/custom_env_and_visibility.py"` | Visibility matrix: public visible to all, owner-only to self, system to L3 only, upper_level to L2+; Microgrid simulation: 5 steps with irradiance-varying solar output |
| `mixed_action_spaces.py` | `python "examples/7. advanced_patterns/mixed_action_spaces.py"` | Continuous generator actions scale correctly; Discrete transformer taps select from 11 positions; Mixed actions combine continuous + discrete; Heterogeneous agents step together in env |

---

## 6. Jupyter Notebooks - Framework

### 6.1 `examples/notebooks/action_passing_tutorial.ipynb`

**Run command:**
```bash
jupyter nbconvert --to notebook --execute examples/notebooks/action_passing_tutorial.ipynb
```

| Criteria | How to Verify |
|----------|---------------|
| All cells execute | No cell errors during execution |
| Training completes | 50 episodes run, returns improve over time |
| Power convergence | Average power moves toward 0 from initial random values |
| Event-driven completes | 100s simulation finishes; message delivery events logged |
| Plots render | Training return plot and power convergence plot generated |

---

### 6.2 `examples/notebooks/ctde_event_driven_tutorial.ipynb`

**Run command:**
```bash
jupyter nbconvert --to notebook --execute examples/notebooks/ctde_event_driven_tutorial.ipynb
```

| Criteria | How to Verify |
|----------|---------------|
| All cells execute | No cell errors |
| CTDE training | 100 episodes; initial return ~ -70, final ~ -20 (improvement) |
| SOC increases | Initial SOC=0.5, final SOC>0.7 (batteries learn to charge) |
| Event-driven | 300s simulation; tick results show battery rewards ~0.03-0.04 |
| Mode comparison | Both synchronous and event-driven produce valid outputs |

---

## 7. Jupyter Notebooks - Power Grid Tutorials

### 7.1 `case_studies/power/tutorials/01_features_and_state.ipynb`

| Criteria | How to Verify |
|----------|---------------|
| All cells execute | No errors |
| Features register | 5 features auto-registered and retrievable by name |
| Visibility rules | Owner sees own features; peer sees only public; coordinator sees upper_level |
| State serialization | Round-trip serialization/deserialization preserves data |

### 7.2 `case_studies/power/tutorials/02_building_agents.ipynb`

| Criteria | How to Verify |
|----------|---------------|
| All cells execute | No errors |
| Agent hierarchy | 3-level system with 2 coordinators and 4 field agents built |
| Action spaces | Battery: Box(-1,1,(1,)); Generator: Box(0,1,(1,)) |
| Physics correct | Battery SOC updates: 0.5 -> 0.625 after charge action |

### 7.3 `case_studies/power/tutorials/03_building_environment.ipynb`

| Criteria | How to Verify |
|----------|---------------|
| All cells execute | No errors |
| Env created | 11 registered agents (including proxy) |
| Reset works | Returns observations for all agents |
| 5 steps complete | Rewards are finite; terminated/truncated tracked correctly |

### 7.4 `case_studies/power/tutorials/04_training_with_rllib.ipynb`

| Criteria | How to Verify |
|----------|---------------|
| All cells execute | No errors |
| Hierarchy | 3 coordinators with heterogeneous obs_dims (e.g., 7, 9, 12) |
| Training | 30 episodes; returns improve over training |
| Event-driven eval | 300s simulation; ~70 events processed, ~12 agent ticks |
| Action decomposition | Joint actions correctly split to per-device via VerticalProtocol |

### 7.5 `case_studies/power/tutorials/05_event_driven_testing.ipynb`

| Criteria | How to Verify |
|----------|---------------|
| All cells execute | No errors |
| Tick configs | Deterministic: exact intervals; Jittered: 10% Gaussian variability |
| Event cascade | Hierarchy respected: L3 ticks -> L2 ticks -> L1 ticks |
| Timeline plots | Agent tick timelines and event type plots render correctly |
| Mode equivalence | Same policy produces valid output in both sync and async modes |

### 7.6 `case_studies/power/tutorials/06_custom_protocols.ipynb`

| Criteria | How to Verify |
|----------|---------------|
| All cells execute | No errors |
| Proportional protocol | Distributes action proportionally to weights (e.g., 3:1 -> 0.6/0.2) |
| Price signal protocol | All agents receive broadcast price in message |
| Hybrid protocol | Combines price message + proportional action in single `coordinate()` |
| Protocol swapping | Same agent runs with different protocols without code changes |

---

## Running Everything

### Full pytest suite
```bash
source .venv/bin/activate
pytest tests/ case_studies/power/tests/ -v
```

### All integration tests (manual scripts)
```bash
source .venv/bin/activate
python tests/integration/test_action_passing.py
python tests/integration/test_e2e.py
python tests/integration/test_maddpg_action_passing.py
python tests/integration/test_qmix_action_passing.py
python tests/integration/test_rllib_action_passing.py
```

### All case study scripts
```bash
source .venv/bin/activate
python case_studies/power/ev_public_charging_case/run_single_station_rollout.py
python case_studies/power/ev_public_charging_case/train_rllib.py
python case_studies/power/ev_public_charging_case/run_event_driven.py
```

### All example scripts
```bash
source .venv/bin/activate
python "examples/1. starter/case1.py"
python "examples/1. starter/case2.py"
python "examples/2. core_abstractions/actions_and_spaces.py"
python "examples/2. core_abstractions/features_and_visibility.py"
python "examples/2. core_abstractions/observations_and_state.py"
python "examples/3. building_environments/custom_heron_env.py"
python "examples/3. building_environments/env_builder_patterns.py"
python "examples/3. building_environments/simple_env_quickstart.py"
python "examples/4. protocols_and_coordination/custom_protocol.py"
python "examples/4. protocols_and_coordination/horizontal_state_sharing.py"
python "examples/4. protocols_and_coordination/vertical_action_decomposition.py"
python "examples/5. training_algorithms/policy_and_training.py"
python "examples/5. training_algorithms/rllib_integration.py"
python "examples/6. event_driven_simulation/dual_mode_execution.py"
python "examples/6. event_driven_simulation/schedule_config_and_scheduling.py"
python "examples/7. advanced_patterns/custom_env_and_visibility.py"
python "examples/7. advanced_patterns/mixed_action_spaces.py"
```

### All notebooks
```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute "examples/notebooks/action_passing_tutorial.ipynb"
jupyter nbconvert --to notebook --execute "examples/notebooks/ctde_event_driven_tutorial.ipynb"
jupyter nbconvert --to notebook --execute "case_studies/power/tutorials/01_features_and_state.ipynb"
jupyter nbconvert --to notebook --execute "case_studies/power/tutorials/02_building_agents.ipynb"
jupyter nbconvert --to notebook --execute "case_studies/power/tutorials/03_building_environment.ipynb"
jupyter nbconvert --to notebook --execute "case_studies/power/tutorials/04_training_with_rllib.ipynb"
jupyter nbconvert --to notebook --execute "case_studies/power/tutorials/05_event_driven_testing.ipynb"
jupyter nbconvert --to notebook --execute "case_studies/power/tutorials/06_custom_protocols.ipynb"
```

---

## Priority Tiers

When time is limited, run tests in this order:

### Tier 1 - Must Pass (core correctness)
1. `pytest tests/test_env_builder.py -v`
2. `python tests/integration/test_action_passing.py`
3. `python tests/integration/test_e2e.py`
4. `python tests/integration/test_rllib_action_passing.py`
5. `python case_studies/power/tests/test_hierarchical_env.py`

### Tier 2 - Should Pass (algorithm coverage)
6. `python tests/integration/test_maddpg_action_passing.py`
7. `python tests/integration/test_qmix_action_passing.py`
8. `python "examples/5. training_algorithms/policy_and_training.py"`
9. `python "examples/5. training_algorithms/rllib_integration.py"`

### Tier 3 - Should Pass (API surface)
10. All Level 2-4 example scripts (core abstractions, environments, protocols)
11. All Level 6-7 example scripts (event-driven, advanced patterns)

### Tier 4 - Nice to Have (tutorials & demos)
12. Case study run scripts
13. Starter examples
14. All Jupyter notebooks
