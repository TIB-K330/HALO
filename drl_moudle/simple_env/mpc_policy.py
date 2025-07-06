import os
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple
import threading

import gym

import torch as th
import rospy
from nav_msgs.msg import Path as NavPath
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from ocp_planner.srv import OcpLocalPlann, OcpLocalPlannRequest
from ocp_planner.msg import HumanState as HumanStateMsg
from ocp_planner.msg import ObstacleState as ObstacleStateMsg
from ocp_planner.msg import PolyState, Point
from simple_env.agent import ActionDiff
from simple_env.info import *
from modules.envs.simple_env import DiscreteEnv
from algorithms.mpc_ppo import MpcPPO 
from algorithms.graph_ppo import joint_state_as_graph
from crowd_sim.envs.utils.state import JointState, FullState, ObstacleState, WallState, ObservableState
from crowd_sim.envs.utils.utils import point_to_segment_dist, counterclockwise, point_in_poly, map_action_to_goal
import ocp_planner_py

class MpcPolicy(object):
    def __init__(self,
                 env: gym.Env,
                 # seed: int = 99,
                 use_ros: bool = False) -> None:
        self.name = 'MPC'
        self.trainable = False
        self.env: gym.Env = env
        self.action_space = None
        self.action_index: int = 0
        self.use_ros = use_ros
        self.use_action_mask: bool = False
        # self.seed = seed
        self.smooth_path: np.ndarray = None
        self.global_path_lock = threading.Lock()
        self.gamma = 0.0
        self.action_prob = 0.0

        if self.use_ros:
            self.ocp_planner = rospy.ServiceProxy('/ocp_plann', OcpLocalPlann)
            rospy.Subscriber("/a_star_path", NavPath, self.path_callback)
        else:
            self.ocp_planner = ocp_planner_py.OcpPlanner()
        self.MPC_NP = 10
        self.nav_goal = np.zeros([2])

    def predict(self,
                observation: JointState,
                deterministic: bool = False,):

        self.nav_goal[0] = observation.robot_state.gx
        self.nav_goal[1] = observation.robot_state.gy
        if self.use_ros:
            ocp_output, _, _ = self.run_solver_ros(observation)
        else:
            ocp_output, _, _ = self.run_solver(observation)
        
        return np.expand_dims(ocp_output, axis=0)

    def run_solver_ros(self, observation: JointState) -> Tuple[np.ndarray, np.ndarray, bool]:
        req = OcpLocalPlannRequest()
        req.ob.robot_state.pose.x = observation.robot_state.px
        req.ob.robot_state.pose.y = observation.robot_state.py
        req.ob.robot_state.pose.theta = self.wrap_pi(observation.robot_state.theta)
        req.ob.robot_state.vl = observation.robot_state.vx
        req.ob.robot_state.vr = observation.robot_state.vy
        req.ob.robot_state.radius = observation.robot_state.radius
        req.ob.robot_state.gx = observation.robot_state.gx
        req.ob.robot_state.gy = observation.robot_state.gy

        if len(observation.human_states) == len(observation.human_pred_states):
            for human, pred_path in zip(observation.human_states, observation.human_pred_states):
                human_state = HumanStateMsg()
                human_state.px = human.px
                human_state.py = human.py
                human_state.vx = human.vx
                human_state.vy = human.vy
                human_state.radius = human.radius
                # print("===============================")
                # print(f"orig x: {human.px}")
                # print(f"orig y: {human.py}")
                human_pred_path = list()
                trajectory = NavPath()
                trajectory.header = Header()
                for x, y in zip(pred_path[0], pred_path[1]):
                    pose = PoseStamped()
                    pose.header = trajectory.header
                    pose.pose.position.x = x
                    pose.pose.position.y = y
                    pose.pose.position.z = 0.0
                    pose.pose.orientation.x = 0.0
                    pose.pose.orientation.y = 0.0
                    pose.pose.orientation.z = 0.0
                    pose.pose.orientation.w = 1.0
                    trajectory.poses.append(pose)
                # print("===============================")

                human_pred_path.append(trajectory)
                human_state.trajectory = human_pred_path
                # for
                req.ob.human_states.append(human_state)
        else:
            for hum in observation.human_states:
                hum_msg = HumanStateMsg(hum.px, hum.py, hum.vx, hum.vy, hum.radius, [])
                req.ob.human_states.append(hum_msg)

        for obst in observation.obstacle_states:
            obst_msg = ObstacleStateMsg(obst.px, obst.py, obst.radius)
            req.ob.obstacle_states.append(obst_msg)

        # ignore the front edges that do NOT belong to the central polygon obstacle
        if len(observation.wall_states) > 2:
            poly_edges = observation.wall_states[2:]
            poly_state = PolyState()

            for edge in poly_edges:
                pt = Point()
                pt.x = edge.sx
                pt.y = edge.sy
                poly_state.vertices.append(pt)

            poly_state.vertices = poly_state.vertices[::-1]  # ccw2cw
            poly_state.is_clockwise = True
            req.ob.poly_states.append(poly_state)

        req.sub_goal.x = observation.robot_state.gx
        req.sub_goal.y = observation.robot_state.gy
        # print("call call call call")
        # print("poly: ", req.ob.poly_states)
        res = self.ocp_planner.call(req)  # call ocp_planner to plan a local optimal trajectory
        control_vars = np.zeros((10, 2), dtype=float)
        if len(res.control_vars) != 0:
            for i, var in enumerate(res.control_vars):
                control_vars[i][0] = var.al
                control_vars[i][1] = var.ar

        self.env.robot.cal_trajectory_rk2(control_vars)
        return np.array((res.al, res.ar)), control_vars, res.success

    def run_solver(self, observation: JointState) -> Tuple[np.ndarray, np.ndarray, bool]:

        input = ocp_planner_py.MPCInputForPython()
        input.ob.robot.gx = observation.robot_state.gx
        input.ob.robot.gy = observation.robot_state.gy
        input.ob.robot.px = observation.robot_state.px
        input.ob.robot.py = observation.robot_state.py
        input.ob.robot.v = 0.5 * (observation.robot_state.vx + observation.robot_state.vy)
        input.ob.robot.yaw = self.wrap_pi(observation.robot_state.theta)
        input.ob.robot.radius = observation.robot_state.radius
        input.ob.robot.yaw_rate = (
                0.5 * (observation.robot_state.vy - observation.robot_state.vx) / observation.robot_state.radius)
        # input.ob.robot.v_pref = observation.robot_state.v_pref
        # input.ob.robot.v_pref = 1.0

        hums = list()
        if len(observation.human_states) == len(observation.human_pred_states):
            for human, pred_path in zip(observation.human_states, observation.human_pred_states):
                human_state = ocp_planner_py.HumanState()
                human_state.px = human.px
                human_state.py = human.py
                human_state.vx = human.vx
                human_state.vy = human.vy
                human_state.radius = human.radius
                human_pred_path = list()
                trajectory = list()
                for x, y in zip(pred_path[0], pred_path[1]):
                    pt = ocp_planner_py.Point()
                    pt.x = x
                    pt.y = y
                    pt.v = 1.0
                    trajectory.append(pt)

                human_pred_path.append(trajectory)
                human_state.pred_trajectorys = human_pred_path
                hums.append(human_state)
        else:
            for human in observation.human_states:
                human_state = ocp_planner_py.HumanState()
                # print("raw_p: ", human_state.px)
                human_state.px = human.px
                human_state.py = human.py
                human_state.vx = human.vx
                human_state.vy = human.vy
                human_state.radius = human.radius

                hums.append(human_state)
            # print("after_p: ", human_state.px)

        input.ob.hum = hums

        obst = list()
        # print(type(input.ob.obst))
        for obstacle in observation.obstacle_states:
            obstacle_state = ocp_planner_py.ObstacleState()
            obstacle_state.px = obstacle.px
            obstacle_state.py = obstacle.py
            obstacle_state.radius = obstacle.radius
            obst.append(obstacle_state)
        input.ob.obst = obst

        vertices = list()
        # ignore the front edges that do NOT belong to the centeral polygon obstacle
        if len(observation.wall_states) > 2:
            poly_edges = observation.wall_states[2:]

            for edge in poly_edges:
                pt = ocp_planner_py.Point()
                pt.x = edge.sx
                pt.y = edge.sy
                pt.v = 0.0
                vertices.append(pt)
        vertices = vertices[::-1]  # ccw2cw
        input.ob.rect.vertices = vertices

        input.sub_goal.x = observation.robot_state.gx
        input.sub_goal.y = observation.robot_state.gy
        input.sub_goal.v = 0.0

        input.valid = True

        ans = self.ocp_planner.run_mpc_solver(input)  # call ocp_planner to plan a local optimal trajectory
        if len(ans.astar_path) != 0:
            astar_path = np.zeros([len(ans.astar_path), 3])
            for i, pt in enumerate(ans.astar_path):
                astar_path[i, 0] = pt.x
                astar_path[i, 1] = pt.y
            self.env.set_astar_path(astar_path)
        else:
            print("astar search fail")

        if len(ans.st_path) != 0:
            st_path = np.zeros([len(ans.st_path), 3])
            for i, pt in enumerate(ans.st_path):
                st_path[i, 0] = pt.x
                st_path[i, 1] = pt.y
            self.env.set_st_path(st_path)
        else:
            print("st search fail")
        
        control_vars = np.zeros((10, 2), dtype=float)
        if len(ans.control_vars) != 0:
            for i, var in enumerate(ans.control_vars):
                control_vars[i][0] = var.al
                control_vars[i][1] = var.ar

        self.env.robot.cal_trajectory_rk2(control_vars)
        # self.env.robot.cal_trajectory_eular(control_vars)
        return np.array((ans.al, ans.ar)), control_vars, ans.success

    def path_callback(self, msg: NavPath):
        self.global_path_lock.acquire()
        n = len(msg.poses)
        # if n % 4 != 0:
        #     print("path and closest_point is wrong")
        self.smooth_path = np.zeros([n, 3])

        for i in range(n):
            self.smooth_path[i, 0] = msg.poses[i].pose.position.x
            self.smooth_path[i, 1] = msg.poses[i].pose.position.y
            self.smooth_path[i, 2] = msg.poses[i].pose.position.z
        self.env.set_refline_search_path(self.smooth_path)
        self.global_path_lock.release()

    def get_astar_path(self) -> np.ndarray:
        if self.smooth_path is None:
            return np.zeros([1, 3])

        return np.copy(self.smooth_path)

    
    @staticmethod
    def wrap_pi(x: float) -> float:
        x = (x + np.pi) % (2 * np.pi)
        if x < 0:
            x = x + 2 * np.pi
        return x - np.pi

