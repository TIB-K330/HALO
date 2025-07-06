from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
import numpy as np


class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.id = None
        self.reach_count: int = 0
        self.start_pos = []
        self.pred_state_num = 10
        self.prediction_path = None
        self.true_future_path = None

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    # TODO: 多模态预测!!!
    def get_prediction_path(self):
        x_val = []
        y_val = []
        if self.prediction_path:
            x_val = [state.px for state in self.prediction_path]
            y_val = [state.py for state in self.prediction_path]
        return x_val, y_val

    def get_true_path(self):
        x_val = []
        y_val = []
        if self.true_future_path:
            x_val = [state.px for state in self.true_future_path]
            y_val = [state.py for state in self.true_future_path]
        return x_val, y_val

    def reset_history_path(self):
        if self.history_path is None:
            self.history_path = np.zeros((2, 8))
        self.history_path[0, :] = self.px
        self.history_path[1, :] = self.py

    def get_history_path(self):
        if self.history_path is None:
            print("human history error")
        return np.copy(self.history_path)