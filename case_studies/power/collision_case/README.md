# Collision Detection Case Study

Multi-agent collision detection experiment for networked microgrids using HERON framework.

## Overview

This case study demonstrates HERON's capabilities for hierarchical multi-agent reinforcement learning with realistic observability constraints, asynchronous execution, and event-driven evaluation.

**System Setup**:
- 3 microgrids (MG1, MG2, MG3) connected to IEEE 34-bus distribution network
- Each microgrid has IEEE 13-bus topology with:
  - Energy Storage System (ESS): ±0.5MW, 2MWh capacity
  - Diesel Generator (DG): 0.5-0.66MW with quadratic cost curves
  - Solar PV: 0.1MW renewable generation
  - Wind Turbine: 0.1MW renewable generation

**Collision Metrics**:
- Overvoltage: Bus voltage > 1.05 p.u.
- Undervoltage: Bus voltage < 0.95 p.u.
- Line overloading: Line loading > 100%
- Device over-rating: Generator/storage exceeding rated capacity
- Power factor violations: PF below minimum threshold

**Research Questions**:
1. Does reward sharing reduce collision frequency?
2. How does asynchronous decision-making affect system safety?
3. What is the impact of observation/action delays on collision?

---

## Experiment Design: 8 Groups

We compare **3 variables** (2³ = 8 combinations):

| Variable | Options | Description |
|----------|---------|-------------|
| **Reward Sharing** | Shared / Independent | Average rewards across MGs vs individual |
| **Async Mode** | Sync / Async_diff | Uniform vs heterogeneous tick rates |
| **Latency** | No-latency / Latency | Zero delays vs realistic delays |

### 8-Group Experiment Matrix

| # | Reward | Async | Latency | Expected Collision |
|---|--------|-------|---------|-------------------|
| 1 | Shared | Sync | No | Lowest (baseline) |
| 2 | Shared | Sync | Yes | Low (slight degradation) |
| 3 | Shared | Async | No | Medium (desync impact) |
| 4 | Shared | Async | Yes | Medium-high (combined) |
| 5 | Independent | Sync | No | Medium |
| 6 | Independent | Sync | Yes | Medium-high |
| 7 | Independent | Async | No | High |
| 8 | Independent | Async | Yes | Highest (worst case) |

### Variable Details

**Async Mode**:
- `sync`: All MGs tick every step (tick_interval=1.0s)
- `async_diff`: Heterogeneous rates
  - MG1: tick_interval=5s (fast controller)
  - MG2: tick_interval=10s (medium controller)
  - MG3: tick_interval=15s (slow controller)

**Latency**:
- `no-latency`: obs_delay=0, act_delay=0, jitter=0
- `latency`: obs_delay=0.1s, act_delay=0.2s, jitter_ratio=0.1-0.2
  - Note: Delays only affect event-driven evaluation, not training

---

## Installation

```bash
cd PowerGym-main

# Install with power grid and multi-agent support
pip install -e ".[powergrid,multi_agent]"

# Verify dataset exists
ls case_studies/power/collision_case/data2024.pkl
# or
ls ../Collision/data/data2024.pkl
```

**Requirements**:
- Python >= 3.10
- Ray >= 2.40.0
- PandaPower >= 3.1.0
- PyTorch >= 2.0.0

---

## Quick Start

### Single Experiment (10 iterations for testing)

```bash
cd case_studies/power

# Test sync mode with shared reward
python -m collision_case.train_collision \
    --share-reward \
    --async-mode=sync \
    --stop-iters=10 \
    --log-path=test_sync.csv

# Test async mode
python -m collision_case.train_collision \
    --share-reward \
    --async-mode=async_diff \
    --stop-iters=10 \
    --log-path=test_async.csv
```

### Full Training (300 iterations)

```bash
# Group 1: Shared + Sync + No-latency (baseline)
python -m collision_case.train_collision \
    --share-reward \
    --async-mode=sync \
    --stop-iters=300 \
    --log-path=results/shared_sync_nolatency.csv

# Group 2: Shared + Sync + Latency
python -m collision_case.train_collision \
    --share-reward \
    --async-mode=sync \
    --enable-latency \
    --stop-iters=300 \
    --log-path=results/shared_sync_latency.csv

# Group 3: Shared + Async + No-latency
python -m collision_case.train_collision \
    --share-reward \
    --async-mode=async_diff \
    --stop-iters=300 \
    --log-path=results/shared_async_nolatency.csv

# Group 4: Shared + Async + Latency
python -m collision_case.train_collision \
    --share-reward \
    --async-mode=async_diff \
    --enable-latency \
    --stop-iters=300 \
    --log-path=results/shared_async_latency.csv

# Groups 5-8: Same combinations without --share-reward
python -m collision_case.train_collision \
    --async-mode=sync \
    --stop-iters=300 \
    --log-path=results/independent_sync_nolatency.csv

python -m collision_case.train_collision \
    --async-mode=sync \
    --enable-latency \
    --stop-iters=300 \
    --log-path=results/independent_sync_latency.csv

python -m collision_case.train_collision \
    --async-mode=async_diff \
    --stop-iters=300 \
    --log-path=results/independent_async_nolatency.csv

python -m collision_case.train_collision \
    --async-mode=async_diff \
    --enable-latency \
    --stop-iters=300 \
    --log-path=results/independent_async_latency.csv
```

### Batch Execution

```bash
cd case_studies/power

# Run all 8 experiments sequentially
chmod +x collision_case/run_8_experiments.sh
nohup ./collision_case/run_8_experiments.sh > run_8groups.log 2>&1 &

# Monitor progress
tail -f run_8groups.log
```

**Expected runtime**: 16-32 hours total (2-4 hours per experiment)

---

## Command-Line Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--algo` | str | RL algorithm (default: PPO) |
| `--num-agents` | int | Number of agents (must be 3) |

### Experiment Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--share-reward` | flag | False | Enable reward sharing across microgrids |
| `--async-mode` | str | sync | `sync` or `async_diff` |
| `--enable-latency` | flag | False | Enable obs/act delays (for eval) |
| `--penalty` | float | 10.0 | Collision penalty weight |
| `--jitter-ratio` | float | 0.1 | Delay randomness ratio (±10%) |

### Training Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--stop-iters` | int | 300 | Training iterations |
| `--episode-steps` | int | 24 | Steps per episode (24 hours) |
| `--dataset-path` | str | auto | Path to data2024.pkl |
| `--log-path` | str | None | CSV log file path |

### RLlib Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-env-runners` | int | 0 | Parallel env workers (0=single process) |
| `--no-tune` | flag | False | Disable Ray Tune (for debugging) |

---

## Output Files

### Training Outputs

Each experiment produces:

1. **CSV Log** (`{log_path}`):
   - Per-timestep collision metrics
   - Rewards and safety violations per MG
   - Format: `timestep,episode,reward_sum,safety_sum,MG1_overvoltage,...`

2. **RLlib Checkpoint** (auto-generated):
   - Saved to: `~/ray_results/{exp_name}/checkpoint_{iter}/`
   - Contains trained policies for MG1, MG2, MG3

3. **Training Log** (if using `run_8_experiments.sh`):
   - Saved to: `results_8groups/{exp_name}.log`
   - Contains stdout/stderr output

### CSV Log Format

```csv
timestep,episode,env_instance,pid,reward_sum,safety_sum,
MG1_overvoltage,MG1_undervoltage,MG1_overloading,MG1_safety_sum,
MG2_overvoltage,MG2_undervoltage,MG2_overloading,MG2_safety_sum,
MG3_overvoltage,MG3_undervoltage,MG3_overloading,MG3_safety_sum,
MG1_q_setpoint_sum,MG2_q_setpoint_sum,MG3_q_setpoint_sum
```

---

## Event-Driven Evaluation (Optional)

Test trained policies under realistic async + delay conditions.

```bash
# Evaluate a checkpoint with event-driven simulation
python -m collision_case.evaluate_latency \
    --checkpoint-dir=~/ray_results/shared_sync_latency/checkpoint_000300 \
    --num-episodes=10 \
    --t-end=24.0 \
    --output=eval_shared_sync_latency.json
```

**Note**: Only evaluate the 4 latency groups to measure real delay impact:
- shared_sync_latency
- shared_async_latency
- independent_sync_latency
- independent_async_latency

---

## Results Analysis

```bash
# Generate comparison plots
python -m collision_case.analyze_results \
    --results-dir=results_8groups \
    --output-dir=results_8groups/plots
```

**Generated plots**:
- `training_curves.png`: Collision frequency over 300 iterations (8 lines)
- `final_collision_comparison.png`: Bar chart comparing final performance
- `async_impact.png`: Box plots showing per-MG collision distribution
- `latency_degradation_table.png`: Event-driven vs training collision

---

## Troubleshooting

### Issue: Dataset not found

**Error**: `FileNotFoundError: No dataset pickle found`

**Solution**:
```bash
# Verify dataset location
ls case_studies/power/collision_case/data2024.pkl
ls ../Collision/data/data2024.pkl

# Or specify explicitly
python -m collision_case.train_collision \
    --dataset-path=/path/to/data2024.pkl \
    ...
```

### Issue: Ray memory overflow

**Error**: `OutOfMemoryError` or Ray workers crashing

**Solution**:
```bash
# Use single-process mode
python -m collision_case.train_collision \
    --num-env-runners=0 \
    ...

# Or shutdown Ray between experiments
# (already handled in run_8_experiments.sh)
```

### Issue: Collision frequency stays at 100%

**Possible causes**:
1. **MG2 capacity bottleneck**: MG2 has smaller DG (0.60MW vs MG1's 0.66MW)
2. **Training not converged**: Increase iterations to 500+
3. **Penalty too low**: Increase `--penalty` from 10.0 to 20.0

**Debug steps**:
```bash
# Check per-MG collision in CSV
cat results/shared_sync_nolatency.csv | \
    awk -F',' '{print $7, $8, $9}' | \
    tail -100 | \
    awk '{s1+=$1; s2+=$2; s3+=$3} END {print s1/100, s2/100, s3/100}'

# This shows average overvoltage per MG in last 100 steps
```

### Issue: Training very slow

**Expected**: ~2-4 hours per 300 iterations

**If slower**:
- Reduce `--num-env-runners` (default 0 is fastest for single env)
- Use `--no-tune` to disable Ray Tune overhead
- Check CPU usage: should be near 100% on one core

---

## Architecture Notes

### HERON Agent Hierarchy

```
SystemAgent (system_agent)
    ├── CoordinatorAgent (MG1)
    │   ├── ESS (MG1_ESS1)
    │   ├── Generator (MG1_DG1)
    │   ├── Generator (MG1_PV1, renewable)
    │   └── Generator (MG1_WT1, renewable)
    ├── CoordinatorAgent (MG2)
    │   └── ... (4 devices)
    └── CoordinatorAgent (MG3)
        └── ... (4 devices)
```

**Exposed to RLlib**: Only 3 coordinators (MG1, MG2, MG3)  
**Device agents**: Managed by coordinators via VerticalProtocol

### Async Implementation

**In sync training** (all 8 groups):
- Agents check `is_active_at(current_step)` based on `tick_interval`
- If `current_step % tick_interval != 0`, agent holds previous action
- Example: MG2 (tick=10) only updates at steps 0, 10, 20, 30, ...

**In event-driven eval** (only latency groups):
- EventScheduler manages discrete events
- AGENT_TICK events scheduled at tick_interval
- OBSERVATION_READY events scheduled after obs_delay
- ACTION_EFFECT events scheduled after act_delay

### Device Configurations

| Device | MG1 | MG2 | MG3 | Unit |
|--------|-----|-----|-----|------|
| DG capacity | 0.66 | 0.60 | 0.50 | MW |
| DG cost a | 100 | 100 | 100 | $/MW² |
| DG cost b | 72.4 | 51.6 | 51.6 | $/MW |
| DG cost c | 0.5011 | 0.4615 | 0.4615 | $ |
| ESS capacity | 2.0 | 2.0 | 2.0 | MWh |
| ESS power | ±0.5 | ±0.5 | ±0.5 | MW |
| PV capacity | 0.1 | 0.1 | 0.1 | MW |
| WT capacity | 0.1 | 0.1 | 0.1 | MW |

**Note**: Heterogeneous DG configs are intentional (represents real-world diversity)

---

## Key Files

| File | Purpose |
|------|---------|
| `train_collision.py` | Main training script with RLlib |
| `system_builder.py` | Creates HERON agent hierarchy |
| `collision_env.py` | HERON environment with collision detection |
| `collision_grid_agent.py` | CollisionGridAgent (coordinator with safety metrics) |
| `collision_rllib_env.py` | RLlib MultiAgentEnv wrapper |
| `collision_features.py` | CollisionMetrics feature definition |
| `collision_network.py` | IEEE 34+13 network topology |
| `run_8_experiments.sh` | Batch script for all 8 groups |
| `evaluate_latency.py` | Event-driven evaluation (to be created) |
| `analyze_results.py` | Results analysis and visualization (to be created) |

---

## Citation

If you use this case study in your research, please cite:

```bibtex
@software{heron_collision_case,
  title={Collision Detection Case Study for HERON Framework},
  author={Your Name},
  year={2026},
  url={https://github.com/Criss-Wang/PowerGym}
}
```

---

## References

- HERON Framework: [PowerGym README](../../README.md)
- PandaPower: https://www.pandapower.org/
- RLlib: https://docs.ray.io/en/latest/rllib/
- IEEE Test Feeders: https://cmte.ieee.org/pes-testfeeders/

---

## Contact

For questions or issues:
- Open an issue: https://github.com/Criss-Wang/PowerGym/issues
- Email: zhenlin.wang.criss@gmail.com
