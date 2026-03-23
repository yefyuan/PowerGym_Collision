Core
====

Core abstractions for state, action, observation, and features.

Action
------

.. py:class:: heron.core.action.Action

   Action container with continuous and discrete components.

   .. py:attribute:: c
      :type: np.ndarray

      Continuous action values.

   .. py:attribute:: d
      :type: np.ndarray

      Discrete action values.

   .. py:method:: set_specs(dim_c: int = 0, dim_d: int = 0, range: tuple = None, n_options: list = None)

      Configure action space specifications.

      :param dim_c: Number of continuous dimensions
      :param dim_d: Number of discrete dimensions
      :param range: Tuple of (low, high) arrays for continuous actions
      :param n_options: List of option counts for discrete actions

   .. py:method:: sample()

      Sample random action within specs.

   .. py:method:: reset()

      Reset action to zeros.

   **Usage:**

   .. code-block:: python

      from heron.core.action import Action
      import numpy as np

      action = Action()
      action.set_specs(
          dim_c=2,
          range=(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
      )

      action.sample()  # Random action
      action.c[:] = [0.5, -0.3]  # Set specific values

Observation
-----------

.. py:class:: heron.core.observation.Observation(local: dict = None, global_: dict = None, messages: list = None, timestamp: float = 0.0)

   Observation container for agent perception.

   :param local: Local observation data
   :param global_: Global observation data (centralized mode)
   :param messages: Received messages (distributed mode)
   :param timestamp: Observation timestamp

   .. py:attribute:: local
      :type: dict

      Local observation features.

   .. py:attribute:: global_
      :type: dict

      Global features (available in centralized mode).

   .. py:attribute:: messages
      :type: list

      Messages from other agents (distributed mode).

   **Usage:**

   .. code-block:: python

      from heron.core.observation import Observation

      obs = Observation(
          local={"state": np.array([1.0, 2.0])},
          global_={"price": np.array([50.0])},
          timestamp=0.0
      )

State
-----

.. py:class:: heron.core.state.State

   Base state container with feature composition.

   .. py:method:: add_feature(name: str, feature: Feature)

      Add a feature to the state.

   .. py:method:: get_feature(name: str) -> Feature

      Get feature by name.

   .. py:method:: vector(visibility_tags: list = None) -> np.ndarray

      Get state as numpy array, filtered by visibility.

   .. py:method:: dim(visibility_tags: list = None) -> int

      Get total dimension of visible features.

.. py:class:: heron.core.state.FieldAgentState

   State for field-level agents.

.. py:class:: heron.core.state.CoordinatorState

   State for coordinator agents with subordinate aggregation.

   **Usage:**

   .. code-block:: python

      from heron.core.state import FieldAgentState

      state = FieldAgentState()
      state.add_feature("temperature", temp_feature)
      state.add_feature("humidity", humidity_feature)

      # Get full state vector
      full_vec = state.vector()

      # Get only coordinator-visible features
      coord_vec = state.vector(visibility_tags=["coordinator"])

Feature
-------

.. py:class:: heron.core.feature.Feature(visibility: list[str] = None)

   Base class for composable features with visibility control.

   :param visibility: List of visibility tags (e.g., ["owner", "coordinator"])

   .. py:attribute:: visibility
      :type: list[str]

      Tags controlling who can observe this feature.

   .. py:method:: vector() -> np.ndarray
      :abstractmethod:

      Return feature as numpy array.

   .. py:method:: dim() -> int
      :abstractmethod:

      Return feature dimension.

   **Default Visibility Tags:**

   - ``owner``: Only the owning agent
   - ``coordinator``: Owner and its coordinator
   - ``system``: System-level agents only
   - ``global``: Everyone

   **Usage:**

   .. code-block:: python

      from heron.core.feature import Feature
      import numpy as np

      class TemperatureFeature(Feature):
          def __init__(self, value: float = 20.0):
              super().__init__(visibility=["owner", "coordinator"])
              self.value = value

          def vector(self) -> np.ndarray:
              return np.array([self.value], dtype=np.float32)

          def dim(self) -> int:
              return 1

Policies
--------

.. py:class:: heron.core.policies.Policy

   Base class for agent policies.

   .. py:method:: select_action(observation: Observation) -> Action
      :abstractmethod:

      Select action given observation.

.. py:class:: heron.core.policies.RandomPolicy(action_space)

   Random action selection policy.

.. py:class:: heron.core.policies.RuleBasedPolicy

   Template for rule-based policies.

   **Usage:**

   .. code-block:: python

      from heron.core.policies import RandomPolicy

      policy = RandomPolicy(action_space=env.action_space)
      action = policy.select_action(observation)
