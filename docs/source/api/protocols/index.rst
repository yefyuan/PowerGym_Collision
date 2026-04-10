Protocols
=========

Coordination protocols for vertical and horizontal agent coordination.

Base Classes
------------

.. py:class:: heron.protocols.base.Protocol

   Base class for coordination protocols.

   .. py:method:: execute(agents: list, **kwargs) -> dict
      :abstractmethod:

      Execute the protocol among agents.

.. py:class:: heron.protocols.base.CommunicationProtocol

   Protocol with message-based communication support.

   .. py:method:: execute_distributed(agents: list, broker: MessageBroker) -> dict
      :async:

      Execute protocol with message passing.

Vertical Protocols
------------------

Protocols for top-down coordination between hierarchy levels.

SetpointProtocol
^^^^^^^^^^^^^^^^

.. py:class:: heron.protocols.vertical.SetpointProtocol

   Direct setpoint control from coordinator to subordinates.

   .. py:method:: execute(coordinator, setpoints: dict) -> dict

      Distribute setpoints to subordinate agents.

      :param coordinator: Coordinator agent
      :param setpoints: Dict mapping subordinate IDs to setpoint values

   **Usage:**

   .. code-block:: python

      from heron.protocols.vertical import SetpointProtocol

      protocol = SetpointProtocol()

      setpoints = {
          "device_1": {"p": 1.0, "q": 0.5},
          "device_2": {"p": 0.8, "q": 0.3},
      }
      result = protocol.execute(coordinator, setpoints)

PriceSignalProtocol
^^^^^^^^^^^^^^^^^^^

.. py:class:: heron.protocols.vertical.PriceSignalProtocol(initial_price: float = 50.0)

   Market-based coordination via price signals.

   :param initial_price: Starting price value

   .. py:attribute:: price
      :type: float

      Current price signal.

   .. py:method:: set_price(price: float)

      Update the price signal.

   .. py:method:: execute(coordinator, price: float = None) -> dict

      Broadcast price to all subordinates.

   **Usage:**

   .. code-block:: python

      from heron.protocols.vertical import PriceSignalProtocol

      protocol = PriceSignalProtocol(initial_price=50.0)

      # System operator sets price based on conditions
      protocol.set_price(65.0)  # High demand period
      result = protocol.execute(coordinator)

Horizontal Protocols
--------------------

Protocols for peer-to-peer coordination between same-level agents.

P2PTradingProtocol
^^^^^^^^^^^^^^^^^^

.. py:class:: heron.protocols.horizontal.P2PTradingProtocol

   Peer-to-peer energy trading protocol.

   .. py:method:: execute(agents: list) -> dict

      Execute trading round among agents.

      :returns: Dict with trades and settlement

   **Usage:**

   .. code-block:: python

      from heron.protocols.horizontal import P2PTradingProtocol

      protocol = P2PTradingProtocol()
      result = protocol.execute([agent1, agent2, agent3])
      # result = {"trades": [...], "total_volume": 1.5}

ConsensusProtocol
^^^^^^^^^^^^^^^^^

.. py:class:: heron.protocols.horizontal.ConsensusProtocol(max_iterations: int = 10, tolerance: float = 0.01)

   Distributed consensus algorithm for agreement.

   :param max_iterations: Maximum consensus rounds
   :param tolerance: Convergence tolerance

   .. py:method:: execute(agents: list) -> dict

      Run consensus until convergence.

      :returns: Dict with consensus value and convergence info

   **Usage:**

   .. code-block:: python

      from heron.protocols.horizontal import ConsensusProtocol

      protocol = ConsensusProtocol(max_iterations=20, tolerance=0.001)
      result = protocol.execute(agents)
      # result = {"consensus_value": 42.5, "iterations": 8, "converged": True}

Protocol Selection Guide
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Protocol
     - Direction
     - Use Case
   * - SetpointProtocol
     - Vertical
     - Direct control of subordinate devices
   * - PriceSignalProtocol
     - Vertical
     - Market-based incentive coordination
   * - P2PTradingProtocol
     - Horizontal
     - Energy/resource trading between peers
   * - ConsensusProtocol
     - Horizontal
     - Agreement on shared values/decisions
