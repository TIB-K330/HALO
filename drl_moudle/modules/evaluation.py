import os.path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import logging

# from modules.utils import joint_state_as_graph

import crowd_sim.envs
# from .envs.actions import ActionDiff
# from .envs.infos import *
from crowd_sim.envs.utils.action import ActionDiff
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import map_action_to_accel


def average(input_list):
    return sum(input_list) / len(input_list) if input_list else 0


def evaluate_policy(
    model,
    env: gym.Env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    phase: str = "val",
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
) -> Tuple[float, float, float, int, float]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    . note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param phase: Validation or test a model
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    from modules.utils import JointState

    rewards = []
    average_returns = []
    success_times = []
    discomfort_nums = []
    ep_num_discoms = []

    success_count = 0
    collision_count = 0
    timeout_count = 0
    discomfort_count = 0

    episode_count = 0
    ob = env.reset(phase=phase)
    robot_state = env.robot.get_full_state()
    joint_state = JointState(robot_state, ob)

    while episode_count < n_eval_episodes:
        # start_time = time.time()
        if (isinstance(model.action_space, gym.spaces.Discrete) and
                (hasattr(model, "run_solver") or hasattr(model, "run_solver_ros"))):
            # action_mask, _ = model.update_action_mask_use_mpc(env.action_mask, joint_state)
            # action = model.predict(joint_state, action_mask=action_mask, deterministic=deterministic)
            action = model.predict(joint_state, action_mask=env.action_mask, deterministic=deterministic)

        else:
            action = model.predict(joint_state, deterministic=deterministic)
        # print("Policy inf time:", time.time() - start_time)
        if (isinstance(model.action_space, gym.spaces.Discrete) and not
        (hasattr(model, "run_solver") or hasattr(model, "run_solver_ros"))):
            left_acc, right_acc = map_action_to_accel(action[0])
            new_ob, reward, done, info = env.step(ActionDiff(left_acc, right_acc))
        else:
            env.set_action_index(model.action_index)
            env.set_actions_prob(model.action_prob)
            left_acc = action[0][0]
            right_acc = action[0][1]
        if render:
            env.render(mode="debug")
        new_ob, reward, done, info = env.step(ActionDiff(left_acc, right_acc))

        new_robot_state = env.robot.get_full_state()
        rewards.append(reward)

        if isinstance(info, Discomfort):
            discomfort_count += 1
            num_discom = info.num
        else:
            num_discom = 0

        ep_num_discoms.append(num_discom)

        if callback is not None:
            callback(locals(), globals())

        if done:
            if isinstance(info, ReachGoal):
                success_count += 1
                success_times.append(env.global_time)
            elif isinstance(info, Collision):
                collision_count += 1
            elif isinstance(info, Timeout):
                timeout_count += 1
                logging.info("Timeout")
            else:
                raise ValueError("Invalid end signal from environment!")

            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(model.gamma, t) * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)

            average_returns.append(average(returns))
            rewards = []

            discomfort_nums.append(sum(ep_num_discoms))
            ep_num_discoms = []

            new_ob = env.reset(phase=phase)
            new_robot_state = env.robot.get_full_state()

            episode_count += 1

        joint_state = JointState(new_robot_state, new_ob)



    success_rate = success_count / n_eval_episodes
    collision_rate = collision_count / n_eval_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else env.time_limit
    logging.info(f"Success rate: {100 * success_rate:.2f}%, Collision rate: {100 * collision_rate:.2f}%")
    logging.info(f"Nav Time: {avg_nav_time:.4f}, DN: {sum(discomfort_nums):.4f}")

    
    return success_rate, collision_rate, avg_nav_time, sum(discomfort_nums), average(average_returns)

def evaluate_mpc(
    model,
    env: crowd_sim.envs.CrowdSim,
    n_eval_episodes: int = 10,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
) -> Tuple[float, float, float, int, float]:

    from modules.utils import JointState

    rewards = []
    average_returns = []
    success_times = []
    discomfort_nums = []
    ep_num_discoms = []

    success_count = 0
    collision_count = 0
    timeout_count = 0
    discomfort_count = 0

    episode_count = 0
    ob = env.reset(phase="test")
    robot_state = env.robot.get_full_state()
    joint_state = JointState(robot_state, ob)

    while episode_count < n_eval_episodes:
        # start_time = time.time()
        action = model.predict(joint_state)
        # print("Policy inf time:", time.time() - start_time)
        new_ob, reward, done, info = env.step(ActionDiff(action[0], action[1]))
        new_robot_state = env.robot.get_full_state()

        rewards.append(reward)

        if isinstance(info, Discomfort):
            discomfort_count += 1
            num_discom = info.num
        else:
            num_discom = 0

        ep_num_discoms.append(num_discom)

        if callback is not None:
            callback(locals(), globals())

        if done:
            if isinstance(info, ReachGoal):
                success_count += 1
                success_times.append(env.global_time)
            elif isinstance(info, Collision):
                collision_count += 1
            elif isinstance(info, Timeout):
                timeout_count += 1
            else:
                raise ValueError("Invalid end signal from environment!")

            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(model.gamma, t) * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)

            average_returns.append(average(returns))
            rewards = []

            discomfort_nums.append(sum(ep_num_discoms))
            ep_num_discoms = []

            new_ob = env.reset(phase="test")
            new_robot_state = env.robot.get_full_state()

            episode_count += 1

        joint_state = JointState(new_robot_state, new_ob)

        if render:
            env.render(mode="debug")

    success_rate = success_count / n_eval_episodes
    collision_rate = collision_count / n_eval_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else env.time_limit

    return success_rate, collision_rate, avg_nav_time, discomfort_count, average(average_returns)


def case_test(
    model,
    env: gym.Env,
    test_case: int = 3,
    video_dir: Optional[str] = None,
) -> None:
    from modules.utils import joint_state_as_graph, JointState

    ob = env.reset(phase="test", test_case=test_case)
    done = False
    robot_state = env.robot.get_full_state()
    # joint_state = joint_state_as_graph(JointState(robot_state, ob), device=model.device)
    joint_state = JointState(robot_state, ob)

    while not done:
        if isinstance(model.action_space, gym.spaces.Discrete) and hasattr(model, "run_solver"):
            action = model.predict(joint_state, action_mask=env.action_mask, deterministic=True)
        else:
            action = model.predict(joint_state, deterministic=True)

        if isinstance(model.action_space, gym.spaces.Discrete) and not hasattr(model, "run_solver"):
            left_acc, right_acc = map_action_to_accel(action[0])
            new_ob, reward, done, info = env.step(ActionDiff(left_acc, right_acc))
        else:
            new_ob, reward, done, info = env.step(ActionDiff(action[0][0], action[0][1]))

        env.render('debug')

        new_robot_state = env.robot.get_full_state()
        # joint_state = joint_state_as_graph(JointState(new_robot_state, new_ob), device=model.device)
        joint_state = JointState(new_robot_state, new_ob)

    if video_dir is not None:
        video_file = os.path.join(video_dir, "proposed")
        video_file = video_file + "_test_" + str(test_case) + ".mp4"
    else:
        video_file = None

    # env.render('video', video_file)
