import numpy as np
from gymnasium.spaces import Dict, Box

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()
tf.config.run_functions_eagerly(True)

class ComplexInputNetwork(TFModelV2):
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

        npc_input = tf.keras.layers.Input(obs['NPC'].shape)
        ego_input = tf.keras.layers.Input(obs['Ego'].shape)
        map_input = tf.keras.layers.Input(obs['Map'].shape)

        map1 = tf.keras.layers.Dense(f)(map_input)
        map2 = tf.math.reduce_max(map1, axis=1, keepdims=True)

        ego1 = tf.keras.layers.Dense(f)(ego_input)

        npc1 = tf.keras.layers.Dense(f)(npc_input)

        concat = tf.keras.layers.Concatenate(axis=-1)([tf.broadcast_to(map2, tf.shape(npc1)), tf.broadcast_to(ego1, tf.shape(npc1)), npc1])
        go_rela_out = tf.keras.layers.Dense(f)(concat)
        self.go_rela_model = tf.keras.Model([npc_input, ego_input, map_input], go_rela_out)

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

