PowerGrid Case Study
====================

Multi-agent microgrid control using HERON with PandaPower integration.

Overview
--------

The PowerGrid case study demonstrates HERON applied to power systems with:

- IEEE 13, 34, 123-bus test networks
- Device models (Generator, ESS, Transformer)
- Multi-agent environments with pre-configured setups
- MAPPO training examples with RLlib

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Description
   * - **Networks**
     - IEEE 13, 34, 123-bus test systems via PandaPower
   * - **Devices**
     - Generator, ESS (Energy Storage), Transformer
   * - **Agents**
     - ``PowerGridAgent`` (coordinator), device agents (field level)
   * - **Features**
     - Electrical (P, Q, V), Storage (SOC), Network metrics
   * - **Setups**
     - Pre-configured environments with config and time series data

.. toctree::
   :maxdepth: 2

   architecture
   devices
   networks
   environments
   examples/index

Installation
------------

.. code-block:: bash

   # Install with power grid support
   pip install -e ".[powergrid]"

   # Or full installation with RL support
   pip install -e ".[all]"

Quick Start
-----------

.. code-block:: python

   from powergrid.envs.multi_agent_microgrids import MultiAgentMicrogrids
   from powergrid.setups.loader import load_setup

   # Load environment configuration from a setup
   env_config = load_setup("ieee34_ieee13")
   env_config.update({
       "centralized": True,
       "max_episode_steps": 24,
       "train": True,
   })
   env = MultiAgentMicrogrids(env_config)
   obs_dict, info = env.reset()

   # Each agent acts independently
   for _ in range(24):
       actions = {agent_id: env.action_spaces[agent_id].sample()
                  for agent_id in env.agents}
       obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)

Running Examples
----------------

.. code-block:: bash

   cd case_studies/power

   # Example 1: Single microgrid
   python examples/01_single_microgrid_basic.py

   # Example 5: MAPPO training
   python examples/05_mappo_training.py --test

See :doc:`examples/index` for detailed example documentation.
