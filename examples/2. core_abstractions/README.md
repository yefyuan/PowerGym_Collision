# Core Abstractions

HERON's data model -- the building blocks every environment is made of.

Before wiring agents, protocols, or training algorithms, you need to understand three things:

1. **Features** declare *what* state exists and *who can see it*.
2. **Actions** define a unified continuous/discrete control interface.
3. **Observations** compose features into what each agent actually perceives.

## Quick Start

```bash
cd "examples/2. core_abstractions"

python features_and_visibility.py    # ~5 sec
python actions_and_spaces.py         # ~5 sec
python observations_and_state.py     # ~5 sec
```

## File Structure

```
2. core_abstractions/
├── features_and_visibility.py   # Feature, visibility rules, is_observable_by()
├── actions_and_spaces.py        # Action specs, scale/unscale, Gymnasium interop
├── observations_and_state.py    # State composition, observation filtering, serialization
└── README.md
```

## What You'll Learn

### features_and_visibility.py

| Concept | What's shown |
|---------|-------------|
| `Feature` subclass | Define observable state fields with `@dataclass(slots=True)` |
| `visibility` class variable | `"public"`, `"owner"`, `"upper_level"`, `"system"` |
| `is_observable_by()` | Who can see a feature given the requester's position in the hierarchy |
| `set_values()` / `reset()` | Update and reset feature fields |
| `vector()` / `to_dict()` | Vectorize or serialize feature data |
| `set_feature_name()` | Instance-level name override for multiple instances of the same class |

**Key takeaway**: Visibility is *declared on the feature*, not coded in the agent. This eliminates an entire class of information-leakage bugs in MARL.

### actions_and_spaces.py

| Concept | What's shown |
|---------|-------------|
| Continuous action | `set_specs(dim_c=2, range=(...))` |
| Discrete action | `set_specs(dim_d=1, ncats=[3])` |
| Mixed action | Both `dim_c` and `dim_d` in one Action |
| `scale()` / `unscale()` | Normalize continuous actions to [-1, 1] for policy output |
| `from_gym_space()` | Create Action from Gymnasium Box / Discrete / Dict |
| `set_values()` formats | Dict, scalar, flat vector, or another Action |
| Utility methods | `vector()`, `scalar()`, `copy()`, `is_valid()` |

**Key takeaway**: One `Action` class handles every agent type. Protocols can split, broadcast, or transform actions without knowing whether they're continuous or discrete.

### observations_and_state.py

| Concept | What's shown |
|---------|-------------|
| `State` from features | Compose `FieldAgentState` from multiple `Feature`s |
| `observed_by()` | Automatic visibility filtering -- returns only features the requestor can see |
| `Observation` structure | `local` (own state) + `global_info` (others' visible state) |
| `vector()` variants | `local_vector()`, `global_vector()`, full `vector()` |
| State updates | `update_feature()`, batch `update()`, `reset()` |
| Serialization | `to_dict()` / `from_dict()` for both Observation and State |

**Key takeaway**: The proxy builds observations automatically using visibility rules. You never hand-code "agent A sees fields X, Y but not Z" -- the framework does it for you.

## Concepts Map

```
Feature          Action                 Observation
  ├── visibility         ├── dim_c (continuous)   ├── local (own features)
  ├── vector()           ├── dim_d (discrete)     ├── global_info (others)
  ├── set_values()       ├── scale/unscale        ├── vector() (flat)
  └── is_observable_by() ├── sample()             └── to_dict/from_dict
                         └── from_gym_space()

         │                       │                       │
         └──── compose into ─────┘                       │
                    │                                    │
                  State                                  │
                    ├── features: Dict[str, Feature]     │
                    ├── observed_by() ───────────────────┘
                    └── update() / reset()
```

## Next Steps

Once you understand features, actions, and observations, move on to:

- **Level 3: Building Environments** -- use these abstractions to construct a full HERON environment.
