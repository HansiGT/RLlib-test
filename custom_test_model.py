import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import (
    SlimConv2d,
)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.annotations import override

torch, nn = try_import_torch()

class Model(TorchModelV2, nn.Module):
    """
    The data flow is as follows:

    `obs` (dict{NPC=Box(-1, 1, shape=(num_agents, 20))},
                Ego=Box(-1, 1, shape=(1, 20)),
                Map=Box(-1, 1, shape=(M, 6))})
                -> `GoRela` -> 'individual_obs' + 'auxiliary input' -> out
    `out` -> action and value heads.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        obs = obs_space.original_space
        f = 128
        self.n = obs['NPC'].shape[0]
        self.outputs_per_agent = int(num_outputs / self.n)

        self.map_dense = nn.Linear(obs['Map'].shape[-1], f)
        self.ego_dense = nn.Linear(obs['Ego'].shape[-1], f)
        self.npc_dense = nn.Linear(obs['NPC'].shape[-1], f)
        self.concat_dense = nn.Linear(3 * f, f)

        # Todo: Fuse Auxiliary Input

        self.preprocessed_obs = None

        # Define the common layers using nn.Sequential
        self.common_layers = nn.Sequential(
            nn.Linear(f, f),
            nn.ReLU()
        )

        # Define the action head
        self.layer_out = nn.Linear(f, self.outputs_per_agent)

        # Define the value head
        self.value_out = nn.Linear(f, 1)

    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict["obs"]["Ego"].shape[0]
        # (print("batch_size: ", batch_size))

        # Map branch
        map1 = self.map_dense(input_dict["obs"]["Map"])
        map2, _ = torch.max(map1, dim=1, keepdim=True)

        # Ego branch
        ego1 = self.ego_dense(input_dict["obs"]["Ego"])

        # NPC branch
        npc1 = self.npc_dense(input_dict["obs"]["NPC"])

        # Concatenate
        map2_broadcasted = map2.expand_as(npc1)
        ego1_broadcasted = ego1.expand_as(npc1)
        concat = torch.cat([map2_broadcasted, ego1_broadcasted, npc1], dim=-1)

        # Preprocessing Output layer
        self.preprocessed_obs = self.concat_dense(concat)

        # Calculate Actions and Values for each agent
        outputs = torch.empty(batch_size, self.n, self.outputs_per_agent)
        self._values = torch.empty(batch_size, self.n)

        for agent_id in range(self.n):
            individual_obs = self.preprocessed_obs[:, agent_id, :]
            x = self.common_layers(individual_obs)

            outputs[:, agent_id] = self.layer_out(x)
            self._values[:, agent_id] = self.value_out(x).squeeze(1)

        return outputs.view(batch_size, self.n * self.outputs_per_agent), state


    def value_function(self):
        assert self._values is not None, "must call forward first!"
        return self._values

