# Level 7: Advanced Patterns

Mixed action spaces, custom environments, and feature visibility.

## Files

| File | Topics |
|------|--------|
| `mixed_action_spaces.py` | Continuous, discrete, and hybrid actions; gym space conversion; scale/unscale; heterogeneous agents |
| `custom_env_and_visibility.py` | Feature visibility modes; `State.observed_by()` filtering; custom `HeronEnv` subclass; `pre_step()` hook |

## Key Concepts

### Action Space Types

```
Continuous   set_specs(dim_c=2, range=(...))        -> Box
Discrete     set_specs(dim_d=1, ncats=[5])          -> Discrete
Multi-disc   set_specs(dim_d=2, ncats=[3, 5])       -> MultiDiscrete
Mixed        set_specs(dim_c=2, dim_d=1, ncats=[3]) -> Dict(c=Box, d=Discrete)
```

### Feature Visibility

| Mode | Who can see |
|------|------------|
| `"public"` | All agents at any level |
| `"owner"` | Only the agent that owns the feature |
| `"system"` | System-level agents (level >= 3) |
| `"upper_level"` | Agents one level above the owner |

Modes combine: `["owner", "upper_level"]` means owner + supervisor.

### Custom HeronEnv

```python
class MyEnv(HeronEnv):
    def run_simulation(self, env_state):         # domain physics
    def global_state_to_env_state(self, gs):     # HERON internal -> your format
    def env_state_to_global_state(self, es):     # your format -> HERON internal
    def pre_step(self):                          # optional per-step hook
```

**Important**: `pre_step()` should store env-level data (profiles, prices) as instance variables, not modify agent state directly. Agent state gets overwritten by `sync_state_from_observed()` during the execute cycle. Flow env-level inputs through `run_simulation()` instead.

## Running

```bash
cd "examples/7. advanced_patterns"
python mixed_action_spaces.py
python custom_env_and_visibility.py
```

## Prerequisites

- [Level 1: Starter](../1.%20starter/) -- agent, feature, env basics
