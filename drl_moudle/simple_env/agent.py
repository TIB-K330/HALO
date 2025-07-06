import numpy as np
from numpy.linalg import norm
from collections import namedtuple
from ocp_planner.msg import ObstacleState, HumanState

from simple_env.orca import ActionXY

ActionDiff = namedtuple('ActionDiff', ['al', 'ar'])


class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta) -> None:
        # for differential model, vx and vy represent v_left and v_right, respectively.
        # v_pref max velocity
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)


class WallState(object):
    def __init__(self, sx, sy, ex, ey) -> None:
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey


class JointState(object):
    def __init__(self, robot_state, observed_states) -> None:
        assert isinstance(robot_state, FullState)
        human_states = observed_states[0]
        obstacle_states = observed_states[1]
        wall_states = observed_states[2]

        for human_state in human_states:
            assert isinstance(human_state, HumanState)

        for obstacle_state in obstacle_states:
            assert isinstance(obstacle_state, ObstacleState)

        for wall_state in wall_states:
            assert isinstance(wall_state, WallState)

        self.robot_state = robot_state
        self.human_states = human_states
        self.obstacle_states = obstacle_states
        self.wall_states = wall_states


class Robot(object):
    def __init__(self) -> None:
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.v_left = None
        self.v_right = None
        self.theta = None
        self.time_step = None
        self.policy = None

    def set(self, px, py, gx, gy, v_left, v_right, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.sx = px
        self.sy = py
        self.gx = gx
        self.gy = gy
        self.v_left = v_left
        self.v_right = v_right
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def set_policy(self, policy):
        self.policy = policy

    def get_full_state(self):
        return FullState(self.px, self.py, self.v_left, self.v_right, self.radius, self.gx, self.gy, self.v_pref,
                         self.theta)

    def get_position(self):
        return self.px, self.py

    def get_goal_position(self):
        return self.gx, self.gy

    def get_start_position(self):
        return self.sx, self.sy

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

    def act(self, ob):
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def check_validity(self, action):
        assert isinstance(action, ActionDiff)

    def compute_position(self, action, dt):
        assert self.time_step == dt
        self.check_validity(action)
        left_acc = action.al
        right_acc = action.ar
        vel_left = self.v_left + left_acc * self.time_step
        vel_right = self.v_right + right_acc * self.time_step

        if np.abs(vel_left) > self.v_pref:
            vel_left = vel_left * self.v_pref / np.abs(vel_left)

        if np.abs(vel_right) > self.v_pref:
            vel_right = vel_right * self.v_pref / np.abs(vel_right)

        t_right = (vel_right - self.v_right) / (right_acc + 1e-9)
        t_left = (vel_left - self.v_left) / (left_acc + 1e-9)
        s_right = (vel_right + self.v_right) * (0.5 * t_right) + vel_right * (self.time_step - t_right)
        s_left = (vel_left + self.v_left) * (0.5 * t_left) + vel_left * (self.time_step - t_left)
        s = (s_right + s_left) * 0.5
        d_theta = (s_right - s_left) / (2 * self.radius)
        s_direction = self.unwarp(self.theta + d_theta * 0.5)
        px_pred = self.px + s * np.cos(s_direction)
        py_pred = self.py + s * np.sin(s_direction)
        return px_pred, py_pred

    def step(self, action):
        left_acc = action.al
        right_acc = action.ar
        vel_left = self.v_left + left_acc * self.time_step
        vel_right = self.v_right + right_acc * self.time_step

        if np.abs(vel_left) > self.v_pref:
            vel_left = vel_left * self.v_pref / np.abs(vel_left)

        if np.abs(vel_right) > self.v_pref:
            vel_right = vel_right * self.v_pref / np.abs(vel_right)

        t_right = (vel_right - self.v_right) / (right_acc + 1e-9)
        t_left = (vel_left - self.v_left) / (left_acc + 1e-9)
        s_right = (vel_right + self.v_right) * (0.5 * t_right) + vel_right * (self.time_step - t_right)
        s_left = (vel_left + self.v_left) * (0.5 * t_left) + vel_left * (self.time_step - t_left)
        s = (s_right + s_left) * 0.5
        d_theta = (s_right - s_left) / (2 * self.radius)
        s_direction = (self.theta + d_theta * 0.5) % (2 * np.pi)
        end_theta = (self.theta + d_theta) % (2 * np.pi)

        end_robot_x = self.px + s * np.cos(s_direction)
        end_robot_y = self.py + s * np.sin(s_direction)
        self.v_left = vel_left
        self.v_right = vel_right
        self.theta = end_theta
        self.px = end_robot_x
        self.py = end_robot_y
        linear_vel = (self.v_right + self.v_left) / 2.0
        vx = linear_vel * np.cos(self.theta)
        vy = linear_vel * np.sin(self.theta)
        self.vx = vx
        self.vy = vy

    @staticmethod
    def unwarp(theta):
        theta = (theta + np.pi) % (2 * np.pi)
        return theta + np.pi if theta < 0 else theta - np.pi 


class Human(object):
    def __init__(self) -> None:
        self.id = None
        self.reach_count = 0
        self.start_pos = []
        self.v_pref = 1.0
        self.radius = 0.3

        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.time_step = None

    def set(self, px, py, gx, gy, vx, vy, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.sx = px
        self.sy = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        
        if radius is not None:
            self.radius = radius
        
        if v_pref is not None:
            self.v_pref = v_pref

    def act(self, ob):
        pass

    def get_observable_state(self):
        return HumanState(self.px, self.py, self.vx, self.vy, self.radius)
    
    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, None)    

    def get_position(self):
        return self.px, self.py

    def get_goal_position(self):
        return self.gx, self.gy

    def check_validity(self, action):
        assert isinstance(action, ActionXY)

    def step(self, action):
        self.check_validity(action)
        self.px += action.vx * self.time_step
        self.py += action.vy * self.time_step
        self.vx = action.vx
        self.vy = action.vy

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius


class Obstacle(object):
    def __init__(self) -> None:
        self.radius = 0.3
        self.px = None
        self.py = None

    def sample_random_attributes(self):
        self.radius = np.clip(np.random.normal(0.25, 0.05), 0.1, 0.4)

    def set(self, px, py, radius=None):
        self.px = px
        self.py = py
        if radius is not None:
            self.radius = radius

    def get_observable_state(self):
        return ObstacleState(self.px, self.py, self.radius)

    def get_position(self):
        return self.px, self.py


class Wall(object):
    def __init__(self) -> None:
        self.sx = None
        self.sy = None
        self.ex = None
        self.ey = None

    def get_observable_state(self):
        return WallState(self.sx, self.sy, self.ex, self.ey)

    def get_position(self):
        pass

    def set_position(self, start_position, end_position):
        self.sx = start_position[0]
        self.sy = start_position[1]
        self.ex = end_position[0]
        self.ey = end_position[1]
        
