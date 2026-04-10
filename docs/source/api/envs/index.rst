Environments
============

Base environment interfaces for HERON.

BaseEnv
-------

.. py:class:: heron.envs.base.BaseEnv

   Abstract base environment class.

   .. py:method:: reset(seed: int = None, options: dict = None) -> tuple
      :abstractmethod:

      Reset environment to initial state.

      :param seed: Random seed
      :param options: Additional options
      :returns: Tuple of (observations, infos)

   .. py:method:: step(actions: dict) -> tuple
      :abstractmethod:

      Execute one environment step.

      :param actions: Dict mapping agent IDs to actions
      :returns: Tuple of (observations, rewards, terminateds, truncateds, infos)

   .. py:attribute:: possible_agents
      :type: list[str]

      List of all possible agent IDs.

   .. py:attribute:: agents
      :type: list[str]

      List of currently active agent IDs.

   .. py:attribute:: observation_spaces
      :type: dict

      Observation space for each agent.

   .. py:attribute:: action_spaces
      :type: dict

      Action space for each agent.

PettingZoo Compatibility
------------------------

HERON environments implement the PettingZoo ``ParallelEnv`` interface:

.. code-block:: python

   from pettingzoo import ParallelEnv

   class MyEnv(ParallelEnv):
       metadata = {"name": "my_env_v0"}

       def __init__(self, config: dict = None):
           self.possible_agents = ["agent_0", "agent_1"]
           self.agents = self.possible_agents.copy()

       def reset(self, seed=None, options=None):
           observations = {agent: self._get_obs(agent) for agent in self.agents}
           infos = {agent: {} for agent in self.agents}
           return observations, infos

       def step(self, actions):
           # Apply actions
           for agent_id, action in actions.items():
               self._apply_action(agent_id, action)

           # Get results
           observations = {agent: self._get_obs(agent) for agent in self.agents}
           rewards = {agent: self._compute_reward(agent) for agent in self.agents}
           terminateds = {agent: self._is_done() for agent in self.agents}
           terminateds["__all__"] = self._is_done()
           truncateds = {agent: False for agent in self.agents}
           truncateds["__all__"] = False
           infos = {agent: {} for agent in self.agents}

           return observations, rewards, terminateds, truncateds, infos

       def observation_space(self, agent):
           return self.observation_spaces[agent]

       def action_space(self, agent):
           return self.action_spaces[agent]

RLlib Integration
-----------------

Wrap HERON environments for RLlib:

.. code-block:: python

   from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
   from ray.tune.registry import register_env

   def env_creator(config):
       env = MyHeronEnv(config)
       return ParallelPettingZooEnv(env)

   register_env("my_heron_env", env_creator)

   # Use in RLlib config
   config = PPOConfig().environment(env="my_heron_env", env_config={...})

Gymnasium Compatibility
-----------------------

For single-agent scenarios or wrapped multi-agent:

.. code-block:: python

   import gymnasium as gym
   from gymnasium import spaces

   class SingleAgentWrapper(gym.Env):
       """Wrap multi-agent env as single-agent."""

       def __init__(self, multi_env, agent_id: str):
           self.env = multi_env
           self.agent_id = agent_id
           self.observation_space = multi_env.observation_space(agent_id)
           self.action_space = multi_env.action_space(agent_id)

       def reset(self, seed=None, options=None):
           obs, info = self.env.reset(seed=seed, options=options)
           return obs[self.agent_id], info.get(self.agent_id, {})

       def step(self, action):
           actions = {self.agent_id: action}
           # Use fixed policy for other agents
           for other in self.env.agents:
               if other != self.agent_id:
                   actions[other] = self.env.action_space(other).sample()

           obs, rewards, terms, truncs, infos = self.env.step(actions)
           return (
               obs[self.agent_id],
               rewards[self.agent_id],
               terms.get(self.agent_id, terms["__all__"]),
               truncs.get(self.agent_id, truncs["__all__"]),
               infos.get(self.agent_id, {})
           )
