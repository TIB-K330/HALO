#!/usr/bin/env python
import argparse
import os
import shutil
import sys
import numpy as np
import torch
import argparse

import gym
import importlib.util
import logging
import torch as th
from sympy import false

from crowd_sim.envs.utils.robot import Robot

from simple_env.mpc_policy import MpcPolicy
from algorithms.mpc_ppo import MpcPPO
from modules.evaluation import evaluate_policy
from modules.policies import ExternalPolicy
from crowd_sim.envs.utils.action import ActionDiff
from crowd_sim.envs.utils.state import JointState

def seed_everything(seed):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # rospy.init_node('ocp_test_node')
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if os.path.exists(args.output_image_dir):
        shutil.rmtree(args.output_image_dir)
        
    os.makedirs(args.output_dir)
    os.makedirs(args.output_image_dir)
    
    # copy config file from ${args.config} to ${args.output_dir} as config.py
    shutil.copy(args.config, os.path.join(args.output_dir, "config.py"))
    args.config = os.path.join(args.output_dir, "config.py")

    # configure logging
    log_file = os.path.join(args.output_dir, "output.log")
    file_handler = logging.FileHandler(log_file, mode='a' if args.resume else 'w')
    # file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    # level = logging.INFO
    logging.basicConfig(
        level=level,
        handlers=[stdout_handler, file_handler],
        format='%(asctime)s, %(levelname)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # load config module of simulation environment
    spec = importlib.util.spec_from_file_location("config", args.config)
    if spec is None:
        parser.error("Config file not found.")

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure environment
    env_config = config.EnvConfig(args.debug)
    env_config.reward.success_reward = args.re_arrival
    env_config.reward.goal_factor = args.goal_weight
    env_config.reward.collision_penalty = args.re_collision
    env_config.reward.discomfort_penalty_factor = args.safe_weight
    env_config.reward.re_theta = args.re_theta
    env_config.reward.re_rvo = args.re_rvo

    logging.info('Current random seed: {}'.format(args.randomseed))
    logging.info('Current safe_weight: {}'.format(args.safe_weight))
    logging.info('Current re_rvo_weight: {}'.format(args.re_rvo))
    logging.info('Current re_theta_weight: {}'.format(args.re_theta))
    logging.info('Current goal_weight: {}'.format(args.goal_weight))
    logging.info('Current re_collision: {}'.format(args.re_collision))
    logging.info('Current re_arrival: {}'.format(args.re_arrival))
    device = th.device('cuda:0' if th.cuda.is_available() and args.gpu else 'cpu')
    logging.info('Using device: %s', device)

    # seed_everything(seed=args.randomseed)
    use_ros = args.use_ros
    use_action_mask = args.use_action_mask
    env = gym.make("CrowdSim-v0")
    env.configure(env_config)
    env.set_phase(10)

    robot = Robot(env_config, "robot")
    robot.time_step = env.time_step
    robot.set_policy(ExternalPolicy())
    env.set_robot(robot)  # action space is determined here
    # action_range = 121
    # env.update_action_range(121)
    action_dim = args.action_dim
    goal_range = (-args.action_range, args.action_range)
    env.num_actions_per_dim = action_dim
    env.goal_coord_range = goal_range
    env.use_ros = use_ros
    env.action_space = gym.spaces.Discrete(action_dim * action_dim)  # temp solution
    env.use_AM = use_action_mask
    env.use_PL = True

    if use_action_mask:
        model = MpcPPO(
            "GraphPolicy",
            env,
            seed=args.randomseed,
            tensorboard_log=args.output_dir,
            device=device,
            action_dim=action_dim,
            goal_coord_range=goal_range,
            use_ros=use_ros,
        )
        try:
            model.set_parameters(os.path.join(args.model_dir, "best_model"), device=device)
        except ValueError as e:
            print(f"Caught an exception: {e}")
    else:
        print("==========a star=================")
        env.use_action_mask = False
        model = MpcPolicy(env, use_ros=use_ros)
    if args.n_eval_episodes > 1:
        success_rate, collision_rate, _, _, ave_return = evaluate_policy(
            model,
            env,
            n_eval_episodes=args.n_eval_episodes,
            render=args.visualize,
            phase='test',
            # callback=[callback, curriculum_callback],
        )
        print(f"Success rate: {100 * success_rate:.2f}%", f"Collision rate: {100 * collision_rate:.2f}%")
    else:
        ob = env.reset(phase='test', for_debug=True, seed=args.randomseed)
        robot_state = env.robot.get_full_state()
        joint_state = JointState(robot_state, ob)
        done = False
        while not done:
            if use_action_mask:
                action = model.predict(joint_state, action_mask=env.action_mask, deterministic=True)
            else:
                action = model.predict(joint_state, deterministic=True)
            left_acc = action[0][0]
            right_acc = action[0][1]
            env.set_action_index(model.action_index)
            env.set_actions_prob(model.action_prob)
            env.render(mode="debug", output_file=args.output_image_dir)
            
            ob, reward, done, info = env.step(ActionDiff(left_acc, right_acc))
            new_robot_state = env.robot.get_full_state()
            joint_state = JointState(new_robot_state, ob)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default='configs/mpc_rl.py')
    parser.add_argument('--output_dir', type=str, default='eval_data/test2')
    parser.add_argument('--output_image_dir', type=str, default='images')

    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=True, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--safe_weight', type=float, default=0.5)
    parser.add_argument('--goal_weight', type=float, default=0.1)
    parser.add_argument('--re_collision', type=float, default=-0.25)
    parser.add_argument('--re_arrival', type=float, default=0.25)
    parser.add_argument('--re_rvo', type=float, default=0.01)
    parser.add_argument('--re_theta', type=float, default=0.01)
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--use_ros', default=False, action='store_true')
    parser.add_argument('--use_action_mask', default=True, action='store_true')
    parser.add_argument('--model_dir', type=str, default='data/model1')
    parser.add_argument('--action_dim', type=int, default=9)
    parser.add_argument('--action_range', type=float, default=2.0)
    parser.add_argument('--randomseed', type=int, default=1421, help="[1100-1599]")
    parser.add_argument('--n_eval_episodes', type=int, default=500)
    sys_args = parser.parse_args()
    main(sys_args)
