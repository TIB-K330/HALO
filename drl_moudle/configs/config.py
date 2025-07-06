"""
Never Modify this file! Always copy the settings you want to change to your local file.
"""

import numpy as np


interval = 500
v_pref = 1.0
rotation_constraint = np.pi / 6
kinematics = 'differential'
human_num = 5
obstacle_num = 3
wall_num = 4


class Config(object):
    def __init__(self):
        pass


class BaseEnvConfig(object):
    env = Config()
    env.time_limit = 30
    env.time_step = 0.25
    env.val_size = 100
    env.test_size = 500
    env.train_size = np.iinfo(np.uint32).max - 2000
    env.randomize_attributes = False
    env.robot_sensor_range = 4

    reward = Config()
    reward.collision_penalty = -1.0
    reward.success_reward = 1.0
    reward.goal_factor = 0.1
    reward.discomfort_penalty_factor = 0.2
    reward.discomfort_dist = 0.2
    reward.re_rvo = 0.05
    reward.re_theta = 0.01

    sim = Config()
    sim.train_val_scenario = 'circle_crossing'
    sim.test_scenario = 'circle_crossing'
    sim.square_width = 10
    sim.circle_radius = 4
    sim.human_num = human_num
    sim.nonstop_human = True
    sim.obstacle_num = obstacle_num
    sim.wall_num = wall_num

    sim.centralized_planning = True

    humans = Config()
    humans.visible = True
    humans.policy = 'orca'
    humans.radius = 0.3
    humans.v_pref = v_pref
    humans.sensor = 'coordinates'

    robot = Config()
    robot.visible = False
    robot.kinematics = kinematics
    robot.policy = 'none'
    robot.radius = 0.3
    robot.v_pref = v_pref
    robot.sensor = 'coordinates'
    robot.rotation_constraint = rotation_constraint

    def __init__(self, debug=False):
        if debug:
            self.env.val_size = 1
            self.env.test_size = 1

