# Building Environments

Three ways to build a HERON environment, from simplest to most flexible.

Pick the one that matches your complexity:

| Approach | When to use | Boilerplate |
|----------|-------------|-------------|
| **SimpleEnv** | Your simulation is a `dict -> dict` function | Minimal |
| **EnvBuilder** | You want batch agent creation, coordinator assignment, auto-coordinators | Low |
| **Custom HeronEnv** | You need a custom state object or non-trivial state bridge | Full control |

## Quick Start

```bash
cd "examples/3. building_environments"

python simple_env_quickstart.py    # ~5 sec
python env_builder_patterns.py     # ~5 sec
python custom_heron_env.py         # ~5 sec
```

## File Structure

```
3. building_environments/
├── simple_env_quickstart.py   # SimpleEnv with a simulation function
├── env_builder_patterns.py    # EnvBuilder fluent API patterns
├── custom_heron_env.py        # HeronEnv subclass with custom state bridge
└── README.md
```

## What You'll Learn

### simple_env_quickstart.py

| Concept | What's shown |
|---------|-------------|
| `SimpleEnv` | Zero-boilerplate environment from agents + simulation function |
| Simulation function | `(agent_states: dict) -> dict` with flat `{aid: {feature: {field: val}}}` |
| Manual agent wiring | Create agents, group under `CoordinatorAgent`, pass to `SimpleEnv` |
| `env.reset()` / `env.step()` | Standard Gymnasium-style interaction loop |

**Key takeaway**: `SimpleEnv` auto-bridges between HERON's internal state and a flat dict your simulation can read/write. You never touch `env_state_to_global_state`.

### env_builder_patterns.py

| Concept | What's shown |
|---------|-------------|
| `add_agents(prefix, cls, count)` | Batch-create agents with auto-generated IDs (`sensor_0`, `sensor_1`, ...) |
| `add_agents(..., coordinator="zone_a")` | Assign agents to a coordinator at creation time |
| Auto-coordinator | Omit `add_coordinator` and the builder wraps all agents automatically |
| `add_agent(id, cls, coordinator=...)` | Single named agent with explicit coordinator assignment |
| `add_system_agent(schedule_config=...)` | Custom system-level timing configuration |

**Key takeaway**: `EnvBuilder` resolves the agent hierarchy for you. Coordinator assignment and auto-coordinators eliminate manual wiring.

### custom_heron_env.py

| Concept | What's shown |
|---------|-------------|
| `HeronEnv` subclass | Full control over the simulation bridge |
| `global_state_to_env_state()` | Convert HERON's internal dict to your domain object (`WaterSystemState`) |
| `run_simulation()` | Your physics/domain logic operating on the custom state |
| `env_state_to_global_state()` | Pack simulation results back into HERON's `{agent_states: ...}` format |
| Two-phase state update | Agent actions update features first, *then* simulation runs, *then* results flow back |

**Key takeaway**: The three-method bridge separates HERON's agent lifecycle from your simulation logic. Your simulation never needs to know about proxies, visibility, or message brokers.

## When to Use What

```
                       ┌─────────────────────────┐
                       │ Is your simulation a     │
                       │ simple dict -> dict      │
                       │ function?                │
                       └────────┬────────────────┘
                           yes  │  no
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
            ┌─────▼─────┐             ┌───────▼──────┐
            │ SimpleEnv  │             │ Custom       │
            │ (or        │             │ HeronEnv     │
            │  EnvBuilder│             │ subclass     │
            │  .simulation())          │              │
            └───────────┘             └──────────────┘
                  │
                  │  Need batch agents
                  │  or auto-coordinators?
                  │
            ┌─────▼──────┐
            │ EnvBuilder  │
            └────────────┘
```

## Data Flow (all three approaches)

```
env.step(actions)
  │
  ▼
SystemAgent.execute()
  ├── 1. Agents apply actions        (apply_action updates features)
  ├── 2. global_state_to_env_state   (HERON dict -> your state)
  ├── 3. run_simulation              (your physics / domain logic)
  ├── 4. env_state_to_global_state   (your state -> HERON dict)
  └── 5. Agents observe new state    (proxy builds visibility-filtered obs)

=> returns (obs, rewards, terminated, truncated, infos)
```

With `SimpleEnv`, steps 2 and 4 are auto-generated. With a custom `HeronEnv` subclass, you implement them yourself.

## Next Steps

Now that you can build environments, learn how agents coordinate:

- **Level 4: Protocols & Coordination** -- vertical action decomposition, horizontal state sharing, and custom protocols.
