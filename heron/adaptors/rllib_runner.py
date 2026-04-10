"""Custom RLlib EnvRunner for HERON environments.

``HeronEnvRunner`` extends RLlib's ``MultiAgentEnvRunner`` with:

* Access to the underlying HERON env for metrics injection
* Proper lifecycle management (calls ``heron_env.close()`` on stop)
* **Event-driven evaluation**: when ``env_config["eval_event_driven"]``
  is set (via ``evaluation_config``), ``sample()`` runs HERON's
  event-driven simulation using the runner's trained RLModules.

Step-based training (default)::

    from heron.adaptors.rllib_runner import HeronEnvRunner

    config = (
        PPOConfig()
        .environment(env=RLlibBasedHeronEnv, env_config={...})
        .env_runners(env_runner_cls=HeronEnvRunner, num_env_runners=2)
        .evaluation(
            evaluation_interval=10,
            evaluation_num_env_runners=0,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_config=HeronEnvRunner.evaluation_config(t_end=100.0),
        )
        .framework("torch")
    )
    algo = config.build()
    algo.train()
    algo.evaluate()  # runs event-driven HERON evaluation via the runner
"""

from typing import Any, Dict, List

import numpy as np
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode


class HeronEnvRunner(MultiAgentEnvRunner):
    """HERON-aware multi-agent environment runner for RLlib.

    Provides access to the underlying HERON env via ``_get_heron_env()``
    and ensures proper cleanup on ``stop()``.

    When the eval runner's ``env_config`` contains ``"eval_event_driven": True``
    (injected via ``evaluation_config=AlgorithmConfig.overrides(...)``),
    ``sample()`` creates a fresh HERON env, bridges the runner's trained
    RLModules to HERON policies, and runs event-driven simulation.
    """

    @staticmethod
    def evaluation_config(
        t_end: float = 100.0,
        seed: int = 100,
    ) -> Any:
        """Return an ``evaluation_config`` object for event-driven HERON evaluation.

        Usage::

            .evaluation(
                evaluation_interval=5,
                evaluation_num_env_runners=0,
                evaluation_duration=1,
                evaluation_duration_unit="episodes",
                evaluation_config=HeronEnvRunner.evaluation_config(t_end=50.0),
            )

        Args:
            t_end: Simulation end time per episode.
            seed: Base seed for ScheduleConfig jitter.
        """
        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

        return AlgorithmConfig.overrides(
            env_config={
                "eval_event_driven": True,
                "eval_t_end": t_end,
                "eval_seed": seed,
            },
        )

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)
        self._heron_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    #  sample() override — event-driven evaluation path
    # ------------------------------------------------------------------

    def sample(
        self,
        *,
        num_timesteps: int = None,
        num_episodes: int = None,
        explore: bool = None,
        random_actions: bool = False,
        force_reset: bool = False,
    ) -> List[MultiAgentEpisode]:
        """Sample episodes; uses event-driven simulation when in eval mode.

        When ``env_config["eval_event_driven"]`` is ``True`` (only present
        in eval runners thanks to ``evaluation_config`` merging), delegates
        to :meth:`_sample_event_driven`.  Otherwise falls through to the
        standard ``MultiAgentEnvRunner.sample()``.
        """
        env_config = self.config.env_config or {}
        if env_config.get("eval_event_driven"):
            return self._sample_event_driven(num_episodes=num_episodes or 1)

        return super().sample(
            num_timesteps=num_timesteps,
            num_episodes=num_episodes,
            explore=explore,
            random_actions=random_actions,
            force_reset=force_reset,
        )

    def _sample_event_driven(
        self, num_episodes: int = 1,
    ) -> List[MultiAgentEpisode]:
        """Run event-driven HERON simulation and return MultiAgentEpisodes.

        Builds one HERON env (schedule_config is applied at construction from
        env_config), bridges the runner's trained RLModules to HERON
        policies, then reuses the env across episodes.  Only jitter RNG
        seeds change per episode via ``reset(jitter_seed=...)``.
        """
        from heron.adaptors.rllib import _build_heron_env
        from heron.scheduling.analysis import EpisodeAnalyzer

        env_config = self.config.env_config
        t_end = env_config.get("eval_t_end", 100.0)
        seed = env_config.get("eval_seed", 100)

        # Agent IDs visible to RLlib (from the RLlibBasedHeronEnv wrapper)
        rllib_agent_ids = list(self.env.unwrapped.possible_agents)

        # Build env once; schedule_config from env_config is applied at construction.
        # Bridge policies (stable across episodes).
        heron_env = _build_heron_env(env_config)
        policies = self._bridge_modules_to_policies(heron_env)
        heron_env.set_agent_policies(policies)

        episodes: List[MultiAgentEpisode] = []

        for ep_idx in range(num_episodes):
            # Reset with per-episode jitter seed (schedule_config set at construction)
            ep_seed = seed + ep_idx
            heron_env.reset(seed=ep_seed, jitter_seed=ep_seed)
            analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
            episode_stats = heron_env.run_event_driven(analyzer, t_end=t_end)

            # 3. Build MultiAgentEpisode from reward history
            reward_history = analyzer.get_reward_history()
            summary = episode_stats.summary()

            ma_episode = self._build_episode_from_event_driven(
                reward_history=reward_history,
                agent_ids=rllib_agent_ids,
            )
            episodes.append(ma_episode)

            # 4. Store HERON-specific metrics
            total_reward = sum(
                sum(r for _, r in rewards)
                for rewards in reward_history.values()
                if rewards
            )
            self._heron_metrics = {
                "episode_reward": total_reward,
                "num_events": summary["num_events"],
                "duration": summary["duration"],
            }

        heron_env.close()

        # Populate done-episodes cache so get_metrics() can process them
        self._done_episodes_for_metrics.extend(episodes)

        # Fire the on_sample_end callback (matches parent sample() contract)
        self._callbacks.on_sample_end(
            env_runner=self,
            metrics_logger=self.metrics,
            samples=episodes,
        )

        return episodes

    # ------------------------------------------------------------------
    #  Module bridging
    # ------------------------------------------------------------------

    def _bridge_modules_to_policies(self, heron_env) -> Dict[str, Any]:
        """Create RLlibModuleBridge policies from the runner's MultiRLModule.

        Maps each HERON agent (with ``action_space``) to the appropriate
        RLModule via the configured ``policy_mapping_fn``, then wraps it
        in a ``RLlibModuleBridge``.
        """
        from heron.adaptors.rllib_module_bridge import RLlibModuleBridge

        agent_ids = [
            aid for aid, ag in heron_env.registered_agents.items()
            if ag.action_space is not None
        ]

        policies = {}
        for aid in agent_ids:
            module_id = self.config.policy_mapping_fn(aid, None)
            if module_id in self.module:
                rl_module = self.module[module_id]
                policies[aid] = RLlibModuleBridge(rl_module, aid)

        return policies

    # ------------------------------------------------------------------
    #  Episode construction from event-driven data
    # ------------------------------------------------------------------

    def _build_episode_from_event_driven(
        self,
        reward_history: Dict[str, list],
        agent_ids: List[str],
    ) -> MultiAgentEpisode:
        """Construct a ``MultiAgentEpisode`` from event-driven reward history.

        Each tick in the reward history becomes one env step.  Observations
        and actions are filled with zeros matching the respective spaces
        (the real data is the reward trajectory).  The final step marks
        all agents as terminated.

        Args:
            reward_history: ``{agent_id: [(timestamp, reward), ...]}``
            agent_ids: Agent IDs visible to RLlib (from the adapter).
        """
        env = self.env.unwrapped

        # Per-agent spaces (same pattern as _new_episode())
        obs_spaces = {
            aid: env.get_observation_space(aid) for aid in agent_ids
        }
        act_spaces = {
            aid: env.get_action_space(aid) for aid in agent_ids
        }

        episode = MultiAgentEpisode(
            observation_space=obs_spaces,
            action_space=act_spaces,
            agent_to_module_mapping_fn=self.config.policy_mapping_fn,
        )

        # Dummy zero obs/actions matching each agent's space
        zero_obs = {
            aid: np.zeros(obs_spaces[aid].shape, dtype=np.float32)
            for aid in agent_ids
        }
        zero_acts = {
            aid: np.zeros(act_spaces[aid].shape, dtype=np.float32)
            for aid in agent_ids
        }

        # Filter reward_history to only RLlib-visible agents
        filtered_rh = {
            aid: reward_history.get(aid, []) for aid in agent_ids
        }
        num_ticks = max(
            (len(rw) for rw in filtered_rh.values()), default=1,
        )
        num_ticks = max(num_ticks, 1)  # at least 1 step

        # Reset step
        episode.add_env_reset(
            observations=zero_obs,
            infos={aid: {} for aid in agent_ids},
        )

        # One env step per tick
        for t in range(num_ticks):
            is_last = (t == num_ticks - 1)

            tick_rewards = {}
            for aid in agent_ids:
                agent_rw = filtered_rh.get(aid, [])
                if t < len(agent_rw):
                    tick_rewards[aid] = agent_rw[t][1]  # (timestamp, reward)
                else:
                    tick_rewards[aid] = 0.0

            terminateds = None
            if is_last:
                terminateds = {aid: True for aid in agent_ids}
                terminateds["__all__"] = True

            episode.add_env_step(
                observations=zero_obs,
                actions=zero_acts,
                rewards=tick_rewards,
                infos={aid: {} for aid in agent_ids},
                terminateds=terminateds,
                truncateds=None,
            )

        return episode

    # ------------------------------------------------------------------
    #  Existing helpers
    # ------------------------------------------------------------------

    def _get_heron_env(self):
        """Return the underlying HERON env (if the wrapper exposes it)."""
        env = self.env
        if hasattr(env, "heron_env"):
            return env.heron_env
        return None

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        if self._heron_metrics:
            metrics["heron"] = self._heron_metrics
        return metrics

    def stop(self) -> None:
        heron_env = self._get_heron_env()
        if heron_env is not None:
            heron_env.close()
        super().stop()
