:html_theme.sidebar_secondary.remove:
:html_theme.sidebar_primary.remove:

.. title:: Welcome to HERON!

.. toctree::
   :hidden:
   :maxdepth: 1

   Getting Started <getting_started>
   Key Concepts <key_concepts>
   User Guide <user_guide/index>
   Case Studies <case_studies/index>
   Developer Guide <developer_guide/index>
   API Reference <api/index>

HERON
=====

**Hierarchical Environments for Realistic Observability in Networks**

A domain-agnostic multi-agent reinforcement learning (MARL) framework for hierarchical control systems.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: sd-text-center

   .. grid-item-card:: Getting Started
      :link: getting_started
      :link-type: doc

      Quick start guide with examples

   .. grid-item-card:: Key Concepts
      :link: key_concepts
      :link-type: doc

      Core abstractions and design principles

Quick Examples
--------------

.. tab-set::

   .. tab-item:: HERON Core

      .. code-block:: python

         from heron.agents import CoordinatorAgent, FieldAgent
         from heron.protocols.vertical import SetpointProtocol
         from heron.messaging.memory import InMemoryBroker

         # Create hierarchical agents
         broker = InMemoryBroker()
         field_agent = FieldAgent(agent_id='device_1', level=1, broker=broker)
         coordinator = CoordinatorAgent(
             agent_id='coordinator_1',
             level=2,
             subordinates=[field_agent],
             protocol=SetpointProtocol(),
             broker=broker
         )

         obs = coordinator.observe(global_state)
         action = coordinator.act(obs)

   .. tab-item:: PowerGrid Case Study

      .. code-block:: python

         from powergrid.envs import MultiAgentMicrogrids

         env = MultiAgentMicrogrids(config={
             'network': 'ieee13',
             'num_microgrids': 2,
             'mode': 'centralized'
         })

         obs, info = env.reset()
         for step in range(96):
             actions = {agent: policy(o) for agent, o in obs.items()}
             obs, rewards, dones, truncs, info = env.step(actions)

   .. tab-item:: Coordination Protocols

      .. code-block:: python

         from heron.protocols.vertical import PriceSignalProtocol
         from heron.protocols.horizontal import ConsensusProtocol

         # Vertical: parent -> subordinate coordination
         vertical = PriceSignalProtocol(initial_price=50.0)

         # Horizontal: peer-to-peer coordination
         horizontal = ConsensusProtocol(max_iterations=10)

   .. tab-item:: RLlib Training

      .. code-block:: python

         from ray.rllib.algorithms.ppo import PPOConfig
         from powergrid.envs import MultiAgentMicrogrids

         config = (
             PPOConfig()
             .environment(MultiAgentMicrogrids)
             .training(lr=3e-4, train_batch_size=4000)
         )
         algo = config.build()
         algo.train()

Key Features
------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Hierarchical Agents
      :link: api/agents/index
      :link-type: doc

      3-level agent hierarchy: System, Coordinator, Field agents

   .. grid-item-card:: Feature-Based State
      :link: api/core/index
      :link-type: doc

      Composable Features with visibility control

   .. grid-item-card:: Coordination Protocols
      :link: api/protocols/index
      :link-type: doc

      Vertical (setpoint, price) and horizontal (P2P, consensus)

   .. grid-item-card:: Message Broker
      :link: api/messaging/index
      :link-type: doc

      Extensible pub/sub for distributed execution

   .. grid-item-card:: Dual Execution Modes
      :link: user_guide/centralized_vs_distributed
      :link-type: doc

      Centralized training, distributed deployment

   .. grid-item-card:: Case Studies
      :link: case_studies/index
      :link-type: doc

      Ready-to-use implementations (PowerGrid, and more)

Architecture
------------

.. code-block:: text

   HERON Framework (Domain-Agnostic)
   +-- heron/
   |   +-- agents/          # Hierarchical agent abstractions
   |   +-- core/            # State, Action, Observation, Feature
   |   +-- protocols/       # Vertical & Horizontal coordination
   |   +-- messaging/       # Message broker interface
   |   +-- envs/            # Base environment interface
   |
   +-- Case Studies
       +-- power/           # PowerGrid case study
           +-- agents/      # GridAgent, DeviceAgent
           +-- features/    # Electrical, Storage features
           +-- networks/    # IEEE test systems
           +-- envs/        # MultiAgentMicrogrids
