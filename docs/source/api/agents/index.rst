Agents
======

Hierarchical agent classes for multi-level control systems.

Agent Base Class
----------------

.. py:class:: heron.agents.base.Agent(agent_id: str, level: int = 1)

   Base class for all HERON agents.

   :param agent_id: Unique identifier for the agent
   :param level: Hierarchy level (1 = leaf, higher = closer to root)

   .. py:attribute:: agent_id
      :type: str

      Unique identifier for the agent.

   .. py:attribute:: level
      :type: int

      Position in hierarchy (1 = field agent, 2 = coordinator, etc.)

   .. py:method:: observe() -> Observation
      :abstractmethod:

      Generate observation from current state.

      :returns: Observation object with local and optional global data

   .. py:method:: act(observation: Observation, **kwargs) -> Action
      :abstractmethod:

      Process observation and return action.

      :param observation: Current observation
      :returns: Action to execute

   .. py:method:: reset(seed: int = None)
      :abstractmethod:

      Reset agent to initial state.

      :param seed: Random seed for reproducibility

Field Agent
-----------

.. py:class:: heron.agents.FieldAgent(agent_id: str, **kwargs)

   Leaf-level agent for local sensing and actuation.

   Inherits from :class:`Agent` with ``level=1``.

   **Usage:**

   .. code-block:: python

      from heron.agents import FieldAgent

      class MySensor(FieldAgent):
          def __init__(self, agent_id: str):
              super().__init__(agent_id)
              self.state = FieldAgentState()

          def observe(self) -> Observation:
              return Observation(local={"state": self.state.vector()})

          def act(self, observation: Observation, **kwargs) -> Action:
              # Local control logic
              return self.action

Coordinator Agent
-----------------

.. py:class:: heron.agents.CoordinatorAgent(agent_id: str, subordinates: dict = None, **kwargs)

   Mid-level agent that manages subordinate agents.

   :param subordinates: Dict mapping IDs to subordinate agents

   .. py:attribute:: subordinates
      :type: dict[str, Agent]

      Dictionary of subordinate agents.

   .. py:method:: aggregate_observations() -> dict

      Collect observations from all subordinates.

   **Usage:**

   .. code-block:: python

      from heron.agents import CoordinatorAgent, FieldAgent

      field1 = MySensor("sensor_1")
      field2 = MySensor("sensor_2")

      coordinator = CoordinatorAgent(
          agent_id="zone_controller",
          subordinates={"sensor_1": field1, "sensor_2": field2}
      )

System Agent
------------

.. py:class:: heron.agents.SystemAgent(agent_id: str, subordinates: dict = None, **kwargs)

   Top-level agent for global coordination.

   Inherits from :class:`CoordinatorAgent` with highest hierarchy level.

   **Usage:**

   .. code-block:: python

      from heron.agents import SystemAgent

      system = SystemAgent(
          agent_id="system_operator",
          subordinates={"zone_1": coordinator1, "zone_2": coordinator2}
      )

Proxy Agent
-----------

.. py:class:: heron.agents.Proxy(agent_id: str, actual_agent: Agent, broker: MessageBroker, upstream_id: str = None)

   Wrapper for message-based distributed execution.

   :param actual_agent: The agent to wrap
   :param broker: Message broker for communication
   :param upstream_id: ID of upstream coordinator (if any)

   .. py:method:: step_distributed()
      :async:

      Execute distributed step with message passing.

   **Usage:**

   .. code-block:: python

      from heron.agents import Proxy
      from heron.messaging.memory import InMemoryBroker

      broker = InMemoryBroker()
      proxy = Proxy(
          agent_id="sensor_1_proxy",
          actual_agent=sensor,
          broker=broker,
          upstream_id="zone_controller"
      )

      await proxy.step_distributed()
