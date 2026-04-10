# Custom Protocols

This guide covers creating custom coordination protocols for HERON.

## Protocol Types

HERON supports two types of protocols:

| Type | Direction | Use Case |
|------|-----------|----------|
| Vertical | Parent → Child | Setpoints, prices, commands |
| Horizontal | Peer ↔ Peer | Trading, consensus, negotiation |

## Creating a Vertical Protocol

Vertical protocols coordinate between hierarchy levels:

```python
from heron.protocols.base import Protocol
from heron.agents.base import Agent

class TargetAllocationProtocol(Protocol):
    """Allocate targets from coordinator to subordinates."""

    def __init__(self, allocation_strategy: str = "proportional"):
        self.allocation_strategy = allocation_strategy

    def execute(self, coordinator: Agent, target: float) -> dict:
        """Allocate target among subordinates.

        Args:
            coordinator: Coordinator agent with subordinates
            target: Total target to allocate

        Returns:
            Dict mapping subordinate IDs to allocated targets
        """
        subordinates = coordinator.subordinates
        n = len(subordinates)

        if self.allocation_strategy == "equal":
            allocations = {sub_id: target / n for sub_id in subordinates}

        elif self.allocation_strategy == "proportional":
            # Allocate based on capacity
            total_capacity = sum(
                sub.state.capacity for sub in subordinates.values()
            )
            allocations = {
                sub_id: target * (sub.state.capacity / total_capacity)
                for sub_id, sub in subordinates.items()
            }

        elif self.allocation_strategy == "priority":
            # Allocate to highest priority first
            sorted_subs = sorted(
                subordinates.items(),
                key=lambda x: x[1].priority,
                reverse=True
            )
            allocations = {}
            remaining = target
            for sub_id, sub in sorted_subs:
                alloc = min(remaining, sub.state.capacity)
                allocations[sub_id] = alloc
                remaining -= alloc

        # Apply allocations
        for sub_id, alloc in allocations.items():
            subordinates[sub_id].set_target(alloc)

        return allocations
```

## Creating a Horizontal Protocol

Horizontal protocols coordinate between peers:

```python
from heron.protocols.base import CommunicationProtocol

class AuctionProtocol(CommunicationProtocol):
    """Double auction protocol for resource trading."""

    def __init__(self, max_rounds: int = 5):
        self.max_rounds = max_rounds
        self.bids = []
        self.asks = []

    def execute(self, agents: list[Agent]) -> dict:
        """Execute double auction among agents.

        Args:
            agents: List of participating agents

        Returns:
            Dict with matched trades
        """
        # Collect bids and asks
        self.bids = []
        self.asks = []

        for agent in agents:
            offer = agent.make_offer()
            if offer["type"] == "bid":
                self.bids.append({
                    "agent_id": agent.agent_id,
                    "quantity": offer["quantity"],
                    "price": offer["price"]
                })
            elif offer["type"] == "ask":
                self.asks.append({
                    "agent_id": agent.agent_id,
                    "quantity": offer["quantity"],
                    "price": offer["price"]
                })

        # Match bids and asks
        trades = self._match_orders()

        # Execute trades
        for trade in trades:
            buyer = next(a for a in agents if a.agent_id == trade["buyer"])
            seller = next(a for a in agents if a.agent_id == trade["seller"])

            buyer.receive_resource(trade["quantity"])
            seller.deliver_resource(trade["quantity"])

        return {"trades": trades, "total_volume": sum(t["quantity"] for t in trades)}

    def _match_orders(self) -> list:
        """Match bids and asks using price-time priority."""
        # Sort bids descending by price, asks ascending
        sorted_bids = sorted(self.bids, key=lambda x: -x["price"])
        sorted_asks = sorted(self.asks, key=lambda x: x["price"])

        trades = []
        bid_idx, ask_idx = 0, 0

        while bid_idx < len(sorted_bids) and ask_idx < len(sorted_asks):
            bid = sorted_bids[bid_idx]
            ask = sorted_asks[ask_idx]

            if bid["price"] >= ask["price"]:
                # Match found
                quantity = min(bid["quantity"], ask["quantity"])
                price = (bid["price"] + ask["price"]) / 2

                trades.append({
                    "buyer": bid["agent_id"],
                    "seller": ask["agent_id"],
                    "quantity": quantity,
                    "price": price
                })

                bid["quantity"] -= quantity
                ask["quantity"] -= quantity

                if bid["quantity"] == 0:
                    bid_idx += 1
                if ask["quantity"] == 0:
                    ask_idx += 1
            else:
                break  # No more matches possible

        return trades
```

## Protocol with Message Broker

For distributed execution, protocols use the message broker:

```python
from heron.protocols.base import CommunicationProtocol
from heron.messaging.base import MessageBroker

class DistributedConsensusProtocol(CommunicationProtocol):
    """Consensus protocol using message passing."""

    def __init__(self, broker: MessageBroker, max_iterations: int = 10, tolerance: float = 0.01):
        self.broker = broker
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    async def execute(self, agents: list[Agent]) -> dict:
        """Execute distributed consensus.

        Args:
            agents: List of participating agents

        Returns:
            Dict with consensus value and convergence info
        """
        # Initialize values
        values = {a.agent_id: a.get_local_value() for a in agents}

        for iteration in range(self.max_iterations):
            # Each agent broadcasts its value
            for agent in agents:
                await self.broker.publish(
                    f"consensus/{agent.agent_id}",
                    {"value": values[agent.agent_id], "iteration": iteration}
                )

            # Each agent receives neighbors' values and updates
            new_values = {}
            for agent in agents:
                neighbor_values = []
                for neighbor_id in agent.neighbors:
                    msg = await self.broker.consume(f"consensus/{neighbor_id}")
                    neighbor_values.append(msg["value"])

                # Update with average
                all_values = [values[agent.agent_id]] + neighbor_values
                new_values[agent.agent_id] = np.mean(all_values)

            # Check convergence
            max_diff = max(
                abs(new_values[aid] - values[aid])
                for aid in values
            )

            values = new_values

            if max_diff < self.tolerance:
                return {
                    "consensus_value": np.mean(list(values.values())),
                    "iterations": iteration + 1,
                    "converged": True
                }

        return {
            "consensus_value": np.mean(list(values.values())),
            "iterations": self.max_iterations,
            "converged": False
        }
```

## Registering Custom Protocols

Register protocols for use in environments:

```python
from heron.utils.registry import ProtocolRegistry

# Register custom protocol
ProtocolRegistry.register("auction", AuctionProtocol)
ProtocolRegistry.register("target_allocation", TargetAllocationProtocol)

# Use in environment config
env_config = {
    "horizontal_protocol": "auction",
    "vertical_protocol": "target_allocation",
}
```

## Best Practices

1. **Separate logic from communication**: Protocol logic should be independent of message transport
2. **Handle failures gracefully**: Timeouts, missing messages, partial participation
3. **Document convergence**: For iterative protocols, document convergence guarantees
4. **Test both modes**: Verify behavior in both centralized and distributed modes
