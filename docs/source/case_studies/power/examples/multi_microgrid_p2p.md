# Example 2: Multi-Microgrid P2P Trading

This example demonstrates multiple microgrids coordinating through peer-to-peer (P2P) trading protocols.

## What You'll Learn

- Creating multi-agent environments
- Using P2PTradingProtocol for horizontal coordination
- Implementing energy trading between microgrids

## Architecture

```
┌─────────────┐     P2P Trading     ┌─────────────┐
│    MG1      │◄──────────────────►│     MG2     │
│ (IEEE 13)   │                     │ (IEEE 13)   │
├─────────────┤                     ├─────────────┤
│ Generator   │                     │ ESS         │
│ ESS         │                     │ Solar       │
└─────────────┘                     └─────────────┘
```

## Code

```python
from powergrid.envs.multi_agent_microgrids import MultiAgentMicrogrids
from heron.protocols.horizontal import P2PTradingProtocol

env_config = {
    "centralized": True,
    "max_episode_steps": 24,
    "horizontal_protocol": P2PTradingProtocol(),
}

env = MultiAgentMicrogrids(env_config)
obs, info = env.reset()

# Multi-agent step
for _ in range(24):
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in env.agents
    }
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
```

## P2P Trading Protocol

The P2PTradingProtocol enables energy exchange between microgrids:

```python
from heron.protocols.horizontal import P2PTradingProtocol

protocol = P2PTradingProtocol()

# Agents can trade excess energy
# - Surplus agents offer energy at a price
# - Deficit agents bid for energy
# - Matching occurs based on price/quantity
```

## Running the Example

```bash
cd case_studies/power
python examples/02_multi_microgrid_p2p.py
```

## Key Concepts

### Horizontal Coordination

Unlike vertical protocols (top-down control), horizontal protocols enable peer coordination:

| Vertical | Horizontal |
|----------|------------|
| SetpointProtocol | P2PTradingProtocol |
| PriceSignalProtocol | ConsensusProtocol |
| Parent → Child | Peer ↔ Peer |

### Energy Trading

```python
# Trading happens during step():
# 1. Each agent determines surplus/deficit
# 2. Surplus agents create sell offers
# 3. Deficit agents create buy bids
# 4. Protocol matches bids and offers
# 5. Energy is transferred between agents
```

## Next Steps

- Try [Price Coordination](price_coordination) for hierarchical control
- See [Distributed Mode](distributed_mode) for realistic communication
