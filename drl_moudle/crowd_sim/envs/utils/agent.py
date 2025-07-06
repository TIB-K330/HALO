import abc
import logging
import numpy as np
from numpy.linalg import norm
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionXY, ActionRot, ActionDiff
from crowd_sim.envs.utils.state import ObservableState, FullState
from crowd_sim.envs.utils.utils import theta_mod


class Agent(object):
    def __init__(self, config, section):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.visible = getattr(config, section).visible
        self.v_pref = getattr(config, section).v_pref
        self.radius = getattr(config, section).radius
        self.policy = policy_factory[getattr(config, section).policy]()
        self.sensor = getattr(config, section).sensor
        self.kinematics = self.policy.kinematics if self.policy is not None else None
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
        self.test_acc_limitation = True
        # todo: 应放在human.py，为了简单实现
        self.history_state_num = 8
        self.history_path = None

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def set_policy(self, policy):
        if self.time_step is None:
            raise ValueError('Time step is None')
        policy.set_time_step(self.time_step)
        self.policy = policy
        self.kinematics = policy.kinematics

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None, v_left=0.0, v_right=0.0):
        self.px = px
        self.py = py
        self.sx = px
        self.sy = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.v_left = v_left
        self.v_right = v_right
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_vx = action.v * np.cos(self.theta)
            next_vy = action.v * np.sin(self.theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        if self.kinematics == 'differential':
            return FullState(self.px, self.py, self.v_left, self.v_right, self.radius, self.gx, self.gy, self.v_pref,
                             self.normalize_angle(self.theta))
        else:
            return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.normalize_angle(self.theta))

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_start_position(self):
        return self.sx, self.sy

    def get_velocity(self):
        if self.kinematics == 'differential' or self.kinematics == 'unicycle':
            return self.v_left, self.v_right
        else:
            return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return
    
    def normalize_angle(self, angle):
        
        return np.arctan2(np.sin(angle), np.cos(angle))

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        elif self.kinematics == 'differential':
            assert isinstance(action, ActionDiff)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_time):
        assert self.time_step == delta_time
        self.check_validity(action)
        if self.kinematics == 'holonomic' :
            px = self.px + action.vx * self.time_step
            py = self.py + action.vy * self.time_step
        elif self.kinematics == 'differential':
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
            s_direction = theta_mod(self.theta + d_theta * 0.5)
            end_theta = theta_mod(self.theta + d_theta)
            end_robot_x = self.px + s * np.cos(s_direction)
            end_robot_y = self.py + s * np.sin(s_direction)
            px = end_robot_x
            py = end_robot_y
        else:
            if self.test_acc_limitation is True:
                sim_num = 5
                dt = self.time_step / sim_num
                px = self.px
                py = self.py
                theta = self.theta
                t_vel = action.v
                t_theta = theta_mod(self.theta + action.r + np.pi) - np.pi
                v = t_vel
                vel_l = self.v_left
                vel_r = self.v_right
                radius = self.radius
                for i in range(sim_num):
                    delta_theta = theta_mod(theta - t_theta + np.pi) - np.pi
                    k = -(4 * delta_theta + np.sin(delta_theta))
                    sim_v = t_vel / (1 + 0.2 * np.abs(k))
                    sim_w = sim_v * k
                    sim_v, sim_w = self.limitation(sim_v, sim_w, vel_l, vel_r, radius, dt)
                    px, py, theta, v, w = self.simulation(px, py, theta, sim_v, sim_w, dt)
                    vel_r = v + w * radius
                    vel_l = v - w * radius
            else:
                theta = self.theta + action.r
                px = self.px + np.cos(theta) * action.v * delta_time
                py = self.py + np.sin(theta) * action.v * delta_time
        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        # pos = self.compute_position(action, self.time_step)
        # self.px, self.py = pos
        if self.kinematics == 'holonomic':
            self.px = self.px + action.vx * self.time_step
            self.py = self.py + action.vy * self.time_step
            self.vx = action.vx
            self.vy = action.vy
        elif self.kinematics == 'differential':
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

            # vel = 0.5 * (vel_left + vel_right)
            # omega = 0.5 * (vel_right - vel_left) / 0.3
            # s = vel * self.time_step
            # d_theta = omega * self.time_step
            # end_theta = (self.theta + d_theta) % (2 * np.pi)
            # s_direction = self.theta % (2 * np.pi)

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
        else:
            if self.test_acc_limitation is True:
                sim_num = 5
                dt = self.time_step / sim_num
                px = self.px
                py = self.py
                theta = self.theta
                t_vel = action.v
                t_theta = (self.theta + action.r + np.pi) % (2 * np.pi) - np.pi
                v = t_vel
                vel_l = self.v_left
                vel_r = self.v_right
                radius = self.radius
                for i in range(sim_num):
                    delta_theta = (theta - t_theta + np.pi) % (2 * np.pi) - np.pi
                    k = -(4 * delta_theta + np.sin(delta_theta))
                    sim_v = t_vel / (1 + 0.2 * np.abs(k))
                    sim_w = sim_v * k
                    sim_v, sim_w = self.limitation(sim_v, sim_w, vel_l, vel_r, radius, dt)
                    px, py, theta, v, w = self.simulation(px, py, theta, sim_v, sim_w, dt)
                    vel_r = v + w * radius
                    vel_l = v - w * radius
                self.px = px
                self.py = py
                self.theta = theta
                self.vx = v * np.cos(self.theta)
                self.vy = v * np.sin(self.theta)
                self.v_left = vel_l
                self.v_right = vel_r
            else:
                self.theta = (self.theta + action.r + np.pi) % (2 * np.pi) - np.pi
                self.vx = action.v * np.cos(self.theta)
                self.vy = action.v * np.sin(self.theta)
                self.px = self.px + self.vx * self.time_step
                self.py = self.py + self.vy * self.time_step
        # self.update_history_path()

    def limitation(self, t_v, t_w, vel_l, vel_r, radius, dt):
        t_vel_r = t_v + t_w * radius
        t_vel_l = t_v - t_w * radius
        if np.abs(t_vel_r) > 1.0:
            k = 1.0/np.abs(t_vel_r)
            t_vel_r = t_vel_r * k
            t_vel_l = t_vel_l * k
        if np.abs(t_vel_l) > 1.0:
            k = 1.0/np.abs(t_vel_l)
            t_vel_l = t_vel_l * k
            t_vel_r = t_vel_r * k

        delta_vel_r = t_vel_r - vel_r
        delta_vel_l = t_vel_l - vel_l
        delta_vel = dt * 1.0
        if np.abs(delta_vel_r) > delta_vel:
            t_vel_r = vel_r + np.abs(delta_vel_r)/delta_vel_r*delta_vel
        if np.abs(delta_vel_l) > delta_vel:
            t_vel_l = vel_l + np.abs(delta_vel_l)/delta_vel_l*delta_vel
        t_v = (t_vel_r + t_vel_l)/2.0
        t_w = (t_vel_r - t_vel_l)/(2.0 * radius)
        return t_v, t_w

    def simulation(self, px, py, theta, v, w, dt):
        next_theta = (theta + w * dt + np.pi) % (2*np.pi) - np.pi
        s = v * dt
        px = px + s * np.cos((next_theta + theta)/2.0)
        py = py + s * np.sin((next_theta + theta)/2.0)
        theta = next_theta
        v = v
        return px, py, theta, v, w

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

    def update_history_path(self):
        if self.history_path is None:
            # 初始化为固定长度的 numpy 数组，避免频繁分配内存
            self.history_path = np.zeros((2, 8))
            self.history_path[0, :] = self.px
            self.history_path[1, :] = self.py
        else:
            # 使用 numpy 的切片和赋值操作进行更新
            self.history_path[0, :-1] = self.history_path[0, 1:]
            self.history_path[0, -1] = self.px
            self.history_path[1, :-1] = self.history_path[1, 1:]
            self.history_path[1, -1] = self.py

