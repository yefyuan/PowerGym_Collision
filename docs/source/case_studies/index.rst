Case Studies
============

HERON includes complete case studies demonstrating the framework applied to real-world domains.

.. toctree::
   :maxdepth: 2

   power/index

Available Case Studies
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Case Study
     - Description
   * - :doc:`power/index`
     - Multi-agent microgrid control with PandaPower integration

Adding New Case Studies
-----------------------

To contribute a new case study:

1. Create directory: ``case_studies/your_domain/``
2. Follow the PowerGrid structure as a template
3. Update ``pyproject.toml`` to include your package
4. Add documentation in ``docs/source/case_studies/``

See the :doc:`../developer_guide/contributing` guide for details.
