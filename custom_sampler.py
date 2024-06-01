import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Union

from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.typing import AgentID, EnvID, EpisodeID, PolicyID, TensorType

logger = logging.getLogger(__name__)

class FullStepSampleCollector(SampleCollector):
    def __init__(self,
                 policy_map: PolicyMap,
                 clip_rewards: Union[bool, float],
                 callbacks: "DefaultCallbacks",
                 multiple_episodes_in_batch: bool = True,
                 rollout_fragment_length: int = 200,
                 count_steps_by: str = "env_steps"):
        super().__init__(policy_map, clip_rewards, callbacks, multiple_episodes_in_batch,
                         rollout_fragment_length, count_steps_by)
        self.current_episode_steps = []
        self.episodes = {}
        self.collected_steps = 0

    def _initialize_episode(self, episode_id):
        if episode_id not in self.episodes:
            self.episodes[episode_id] = {}

    def _store_step(self, episode_id, agent_id, key, value):
        if agent_id not in self.episodes[episode_id]:
            self.episodes[episode_id][agent_id] = {
                "obs": [],
                "infos": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "next_obs": []
            }
        self.episodes[episode_id][agent_id][key].append(value)

    def add_init_obs(self,
                     *,
                     episode: Episode,
                     agent_id: AgentID,
                     policy_id: PolicyID,
                     init_obs: TensorType,
                     init_infos: Optional[Dict[str, TensorType]] = None,
                     t: int = -1) -> None:
        self._initialize_episode(episode.episode_id)
        self._store_step(episode.episode_id, agent_id, "obs", init_obs)
        if init_infos:
            self._store_step(episode.episode_id, agent_id, "infos", init_infos)

    def add_action_reward_next_obs(self,
                                   episode_id: EpisodeID,
                                   agent_id: AgentID,
                                   env_id: EnvID,
                                   policy_id: PolicyID,
                                   agent_done: bool,
                                   values: Dict[str, TensorType]) -> None:
        self._store_step(episode_id, agent_id, "actions", values[SampleBatch.ACTION])
        self._store_step(episode_id, agent_id, "rewards", values[SampleBatch.REWARD])
        self._store_step(episode_id, agent_id, "dones", values[SampleBatch.TERMINATED])
        self._store_step(episode_id, agent_id, "next_obs", values[SampleBatch.NEXT_OBS])

        # Check if all agents have provided their actions for this step
        if all(len(agent["actions"]) == len(agent["next_obs"]) for agent in self.episodes[episode_id].values()):
            self.current_episode_steps.append(self.episodes[episode_id])
            self.collected_steps += 1

    def episode_step(self, episode: Episode) -> None:
        pass

    def total_env_steps(self) -> int:
        return self.collected_steps

    def total_agent_steps(self) -> int:
        return sum(len(agent["actions"]) for episode in self.episodes.values() for agent in episode.values())

    def get_inference_input_dict(self, policy_id: PolicyID) -> Dict[str, TensorType]:
        return {
            SampleBatch.OBS: [agent["obs"][-1] for episode in self.episodes.values() for agent in episode.values()]
        }

    def postprocess_episode(self,
                            episode: Episode,
                            is_done: bool = False,
                            check_dones: bool = False,
                            build: bool = False) -> Optional[MultiAgentBatch]:
        if is_done:
            episode_data = self.episodes.pop(episode.episode_id, None)
            if episode_data:
                batches = {policy_id: [] for policy_id in self.policy_map.keys()}
                for agent_id, data in episode_data.items():
                    policy_id = "shared_policy"  # Assuming a shared policy
                    batches[policy_id].append(SampleBatch({
                        SampleBatch.OBS: data["obs"],
                        SampleBatch.ACTIONS: data["actions"],
                        SampleBatch.REWARDS: data["rewards"],
                        SampleBatch.TERMINATEDS: data["dones"],
                        SampleBatch.NEXT_OBS: data["next_obs"],
                    }))
                return MultiAgentBatch({k: SampleBatch.concat_samples(v) for k, v in batches.items()}, self.total_env_steps())
        return None

    def try_build_truncated_episode_multi_agent_batch(self) -> List[Union[MultiAgentBatch, SampleBatch]]:
        if self.collected_steps >= self.rollout_fragment_length:
            batches = {policy_id: [] for policy_id in self.policy_map.keys()}
            for episode in self.current_episode_steps:
                for agent_id, data in episode.items():
                    policy_id = "shared_policy"  # Assuming a shared policy
                    batches[policy_id].append(SampleBatch({
                        SampleBatch.OBS: data["obs"],
                        SampleBatch.ACTIONS: data["actions"],
                        SampleBatch.REWARDS: data["rewards"],
                        SampleBatch.TERMINATEDS: data["dones"],
                        SampleBatch.NEXT_OBS: data["next_obs"],
                    }))
            self.current_episode_steps = []
            self.collected_steps = 0
            return [MultiAgentBatch({k: SampleBatch.concat_samples(v) for k, v in batches.items()}, self.total_env_steps())]
        return []
