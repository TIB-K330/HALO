import abc
import numpy as np
import torch


class Policy(object):
    def __init__(self):
        """
        Base class for all policies, has an abstract method predict().
        """
        self.trainable = False
        self.phase = None
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None
        # if agent is assumed to know the dynamics of real world
        self.env = None
        self.reward_estimator = None
        self.v_pref = 1

    @abc.abstractmethod
    def configure(self, config):
        return

    def set_phase(self, phase):
        self.phase = phase

    def set_v_pref(self, v_pref):
        self.v_pref = v_pref

    def set_device(self, device):
        self.device = device

    def set_env(self, env):
        self.env = env

    def set_time_step(self, time_step):
        self.time_step = time_step

    def get_model(self):
        return self.model

    def save_model(self, file):
        torch.save(self.model.state_dict(), file)

    def load_model(self, file):
        self.model.load_state_dict(torch.load(file))

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @abc.abstractmethod
    def predict(self, state):
        """
        Policy takes state as input and output an action

        """
        return

    @staticmethod
    def reach_destination(state):
        robot_state = state.robot_state
        if np.linalg.norm((robot_state.py - robot_state.gy, robot_state.px - robot_state.gx)) < robot_state.radius:
            return True
        else:
            return False

    def transform(self, state):
        """
        Take the JointState to tensors

        :param state:
        :return: tensor of shape (# of agent, len(state))
        """
        robot_state_tensor = torch.Tensor([state.robot_state.to_tuple()])
        human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in state.human_states])
        obstacle_states_tensor = torch.Tensor([obstacle_state.to_tuple() for obstacle_state in state.obstacle_states])
        wall_states_tensor = torch.Tensor([wall_state.to_tuple() for wall_state in state.wall_states])
        return robot_state_tensor, human_states_tensor, obstacle_states_tensor, wall_states_tensor

