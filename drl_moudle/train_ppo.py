import argparse
import os.path
import shutil
import sys

import gym
import importlib.util
import logging
import torch as th

from crowd_sim.envs.utils.robot import Robot
from modules.policies import ExternalPolicy

# from algorithms.graph_ppo import GraphPPO
from algorithms.mpc_ppo import MpcPPO
from modules.callbacks import CurriculumCallback, EvalCallback, CheckpointCallback
from modules.evaluation import evaluate_policy
from modules.utils import get_linear_fn

def main(args):
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir)
    # copy config file from ${args.config} to ${args.output_dir} as config.py
    shutil.copy(args.config, os.path.join(args.output_dir, "config.py"))
    args.config = os.path.join(args.output_dir, "config.py")

    # configure logging
    log_file = os.path.join(args.output_dir, "output.log")
    file_handler = logging.FileHandler(log_file, mode='a' if args.resume else 'w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
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

    env = gym.make("CrowdSim-v0")
    env.configure(env_config)
    env.set_phase(0)

    robot = Robot(env_config, "robot")
    robot.time_step = env.time_step
    robot.set_policy(ExternalPolicy())
    env.set_robot(robot)  # action space is determined here
    # action_range = 121
    # env.update_action_range(121)
    action_dim = args.action_dim
    goal_range = (-args.action_range, args.action_range)
    use_ros = args.use_ros
    # temp solution
    env.num_actions_per_dim = action_dim
    env.goal_coord_range = goal_range
    env.use_ros = use_ros
    env.action_space = gym.spaces.Discrete(action_dim * action_dim)
    env.use_AM = args.use_AM
    env.use_PL = args.use_PL
    env.PL_traj_length = args.PL_traj_length
    env.PL_traj_gamma = args.PL_traj_gamma

    lr_schedule = get_linear_fn(2.5e-4, 1.0e-4, 0.5)

    model = MpcPPO(
        "GraphPolicy",
        env,
        learning_rate=lr_schedule,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.001,
        tensorboard_log=args.output_dir,
        seed=args.randomseed,
        device=device,
        action_dim=action_dim,
        goal_coord_range=goal_range,
        use_ros=use_ros,
    )
    curriculum_callback = CurriculumCallback(verbose=0)
    eval_callback = EvalCallback(eval_env=env, n_eval_episodes=100, eval_freq=500,
                                 best_model_save_path=args.output_dir, verbose=1)
    model.learn(int(5e6), callback=[eval_callback, curriculum_callback])
    # checkpt_callback = CheckpointCallback(save_freq=2000, save_path=args.checkpt_dir, name_prefix='PPO')
    # model.learn(int(5e6), callback=[eval_callback, curriculum_callback, checkpt_callback])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument('--config', type=str, default='configs/mpc_rl.py')
    # parser.add_argument('--output_dir', type=str, default='data/output_eval_ipopt_theta_4_yaw_penalize')
    parser.add_argument('--output_dir', type=str, default='train_data/action_mask_eval_traj_3_4')
    
    # parser.add_argument('--output_dir', type=str, default='data/output_test')
    parser.add_argument('--resume', default=True, action='store_true')
    parser.add_argument('--gpu', default=True, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=3)
    parser.add_argument('--safe_weight', type=float, default=0.5)
    parser.add_argument('--goal_weight', type=float, default=0.1)
    parser.add_argument('--re_collision', type=float, default=-0.25)
    parser.add_argument('--re_arrival', type=float, default=0.25)
    parser.add_argument('--re_rvo', type=float, default=0.01)
    parser.add_argument('--re_theta', type=float, default=0.01)
    parser.add_argument('--action_dim', type=int, default=9)
    parser.add_argument('--action_range', type=float, default=2.25)
    parser.add_argument('--use_ros', type=bool, default=False)
    parser.add_argument('--use_AM', type=bool, default=True, help="enable Action Mask")
    parser.add_argument('--use_PL', type=bool, default=True, help="enable Privileged Learning")
    parser.add_argument('--PL_traj_length', type=int, default=4)
    parser.add_argument('--PL_traj_gamma', type=float, default=0.9)
    # parser.add_argument('--checkpt_dir', type=str, default='data/output_eval_ipopt_theta_4_yaw_penalize_pt1')


    sys_args = parser.parse_args()
    main(sys_args)
