Examples
========

Hands-on examples demonstrating PowerGrid capabilities.

.. list-table::
   :header-rows: 1
   :widths: 10 30 60

   * - #
     - Example
     - Description
   * - 1
     - :doc:`single_microgrid`
     - Basic single microgrid with centralized control
   * - 2
     - :doc:`multi_microgrid_p2p`
     - Multi-microgrid environment with P2P trading protocol
   * - 3
     - :doc:`price_coordination`
     - Price signal protocol for hierarchical coordination
   * - 4
     - :doc:`custom_device`
     - Creating custom device agents
   * - 5
     - :doc:`mappo_training`
     - Production-ready MAPPO training with RLlib
   * - 6
     - :doc:`distributed_mode`
     - Distributed execution mode with proxy agents

.. toctree::
   :hidden:
   :maxdepth: 1

   single_microgrid
   multi_microgrid_p2p
   price_coordination
   custom_device
   mappo_training
   distributed_mode

Running Examples
----------------

.. code-block:: bash

   # From project root
   cd case_studies/power

   # Run any example
   python examples/01_single_microgrid_basic.py
   python examples/05_mappo_training.py --test
