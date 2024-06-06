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
                Ego=Box(-1, 1, shape=(20,)),
                Map=Box(-1, 1, shape=(M, 6))})
                -> `GoRela` -> 'individual_obs' + 'auxiliary input' -> out
    `out` -> action and value heads.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        obs = obs_space.original_space
        f = 128
        m = obs['Map'].shape[0]
        self.n = obs['NPC'].shape[0]

        self.map_dense = nn.Linear(obs['Map'].shape, f)
        self.ego_dense = nn.Linear(obs['Ego'].shape, f)
        self.npc_dense = nn.Linear(obs['NPC'].shape, f)
        self.concat_dense = nn.Linear(3 * f, f)

        # Todo: Fuse Auxiliary Input

        self.preprocessed_obs = None

        individual_obs = tf.keras.layers.Input(f)
        layer1 = tf.keras.layers.Dense(f, activation=tf.nn.relu)(individual_obs)

        # Actions and value heads.
        layer_out = tf.keras.layers.Dense(num_outputs)(layer1)

        # Create the value branch model.
        value_out = tf.keras.layers.Dense(1)(layer1)

        self.base_model = tf.keras.Model(individual_obs, [layer_out, value_out])

    def forward(self, input_dict, state, seq_lens):
        print(input_dict["eps_id"])
        if isinstance(input_dict["agent_index"], tf.Tensor):
            print("Agent_ID: ", input_dict["agent_index"])
            agent_id = tf.cast(input_dict["agent_index"][0], tf.int32).numpy()
        else:
            print("Agent_ID: ", input_dict["agent_index"])
            agent_id = int(input_dict["agent_index"][0])

        if agent_id == 0:
            self.preprocessed_obs = self.go_rela_model([input_dict["obs"]["NPC"], input_dict["obs"]["Ego"], input_dict["obs"]["Map"]])
            #print(self.preprocessed_obs)
        agent_obs = self.preprocessed_obs[:, agent_id, :]
        model_out, self._value_out = self.base_model(agent_obs)
        #print("Model_out: ", model_out)
        return model_out, state


    def value_function(self):
        #print("Value_out: ", self._value_out)
        return tf.reshape(self._value_out, [-1])

