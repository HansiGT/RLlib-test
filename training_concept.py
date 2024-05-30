import numpy as np

import gymnasium as gym
import os

import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from custom_test_model import ComplexInputNetwork
from ray.rllib.algorithms.ppo import (PPOConfig, PPOTF2Policy)
from dummy_env import DummyEnv
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

ray.init()
ModelCatalog.register_custom_model("test_model", ComplexInputNetwork)

register_env(
    "dummy", lambda _: DummyEnv()
)
env = DummyEnv()
obs_space = env.observation_space
act_space = env.action_space

# Construct two independent Algorithm configs
teacher_config = (
    PPOConfig()
    .environment("dummy")
    .rl_module(_enable_rl_module_api=False)
    .training(
        _enable_learner_api=False,
        model={"custom_model": "test_model"},
        vf_loss_coeff=0.01
        #train_batch_size=128
    )
    .framework(
        framework="tf2",
        eager_max_retraces=None
    )
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    #.experimental(_disable_initialize_loss_from_dummy_batch=True)
)

policies = {"teacher_policy": (PPOTF2Policy, obs_space, act_space, teacher_config)}


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "teacher_policy"


# Add multi-agent configuration options to both configs and build them.
teacher_config.multi_agent(
    policies=policies,
    policy_mapping_fn=policy_mapping_fn
)
teacher = teacher_config.build()

for i in range(100):
        print("-- Teacher --")
        result_teacher = teacher.train()
        #print(pretty_print(result_teacher))

"""
student_config = (
    PPOConfig()
    .experimental(_enable_new_api_stack=False)
    .environment("carla")
    .framework(args.framework)
    # disable filters, otherwise we would need to synchronize those
    # as well to the other agent
    .rollouts(observation_filter="MeanStdFilter")
    .training(
        model={"vf_share_layers": True},
        vf_loss_coeff=0.01,
        num_sgd_iter=6,
    )
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
)

policies = {
    "teacher_policy": (
        select_policy("Teacher", args.framework),
        obs_space,
        act_space,
        teacher_config,
    ),
    "student_policy": (
        select_policy("Student", args.framework),
        obs_space,
        act_space,
        student_config,
    ),
}


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == 0:
        return "student_policy"
    else:
        return "teacher_policy"


# Add multi-agent configuration options to both configs and build them.
teacher_config.multi_agent(
    policies=policies,
    policy_mapping_fn=policy_mapping_fn,
    policies_to_train=["teacher_policy"],
)
teacher = teacher_config.build()

student_config.multi_agent(
    policies=policies,
    policy_mapping_fn=policy_mapping_fn,
    policies_to_train=["student_policy"],
)
student = student_config.build()

for i in range(args.stop_iters):
    print("== Iteration", i, "==")

    # improve the student policy
    for i in range(student_iterations):
        print("-- Student --")
        result_student = student.train()
        print(pretty_print(result_student))

    # swap weights to synchronize
    teacher.set_weights(student.get_weights(["student_policy"]))

    # improve the teacher policy
    for i in range(teacher_iterations):
        print("-- Teacher --")
        result_teacher = teacher.train()
        print(pretty_print(result_teacher))

    # swap weights to synchronize
    student.set_weights(teacher.get_weights(["teacher_policy"]))
"""
