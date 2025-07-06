import warnings
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
# from venv import logger
import logging
import gym
import numpy as np

from .logger import Logger
from .evaluation import evaluate_policy


class BaseCallback(ABC):
    """
    Base class for callback.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    # The RL model
    # Type hint as string to avoid circular import
    model: Any

    def __init__(self, verbose: int = 0):
        super().__init__()
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        self.num_timesteps = 0  # type: int
        self.verbose = verbose
        self.locals: Dict[str, Any] = {}
        self.globals: Dict[str, Any] = {}
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]

    @property
    def training_env(self) -> gym.Env:
        training_env = self.model.get_env()
        assert (
            training_env is not None
        ), "`model.get_env()` returned None, you must initialize the model with an environment to use callbacks"
        return training_env

    @property
    def logger(self) -> Logger:
        return self.model.logger

    # Type hint as string to avoid circular import
    def init_callback(self, model: Any) -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        # Update num_timesteps in case training was done before
        self.num_timesteps = self.model.num_timesteps
        self._on_training_start()

    def _on_training_start(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    @abstractmethod
    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps

        return self._on_step()

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        pass

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables on sub callbacks.

        :param locals_: the local variables during rollout collection
        """
        pass


class EventCallback(BaseCallback):
    """
    Base class for triggering callback on event.

    :param callback: Callback that will be called
        when an event is triggered.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.callback = callback
        # Give access to the parent
        if callback is not None:
            assert self.callback is not None
            self.callback.parent = self

    def init_callback(self, model: Any) -> None:
        super().init_callback(model)
        if self.callback is not None:
            self.callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        if self.callback is not None:
            self.callback.on_training_start(self.locals, self.globals)

    def _on_event(self) -> bool:
        if self.callback is not None:
            return self.callback.on_step()
        return True

    def _on_step(self) -> bool:
        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback is not None:
            self.callback.update_locals(locals_)


class CallbackList(BaseCallback):
    """
    Class for chaining callbacks.

    :param callbacks: A list of callbacks that will be called
        sequentially.
    """

    def __init__(self, callbacks: List[BaseCallback]):
        super().__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            # Return False (stop training) if at least one callback returns False
            continue_training = callback.on_step() and continue_training
        return continue_training

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        for callback in self.callbacks:
            callback.update_locals(locals_)


class ConvertCallback(BaseCallback):
    """
    Convert functional callback (old-style) to object.

    :param callback:
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]], verbose: int = 0):
        super().__init__(verbose)
        self.callback = callback

    def _on_step(self) -> bool:
        if self.callback is not None:
            return self.callback(self.locals, self.globals)
        return True
    

class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    """

    def __init__(
        self,
        eval_env: gym.Env,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,  # (episode)
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_ave_return = -np.inf
        self.last_ave_return = -np.inf
        self.deterministic = deterministic
        self.render = render

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                "Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        pass

    def _on_step(self) -> bool:
        if not self.locals['done']:
            return True

        continue_training = True
        episode_num = self.locals['self']._episode_num + 1

        if self.eval_freq > 0 and episode_num % self.eval_freq == 0:
            if self.verbose >= 1:
                print(f"VAL after {episode_num} episodes of environment interaction")
                logging.info(f"VAL after {episode_num} episodes of environment interaction")
                

            success_rate, collision_rate, _, _, ave_return = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                pass

            self.last_ave_return = float(ave_return)
            self.logger.record("eval/ave_return", float(ave_return))

            if self.verbose >= 1:
                print(f"Success rate: {100 * success_rate:.2f}%", f"Collision rate: {100 * collision_rate:.2f}%")
                logging.info(f"Success rate: {100 * success_rate:.2f}%, Collision rate: {100 * collision_rate:.2f}%")

            self.logger.record("eval/success_rate", success_rate)
            self.logger.dump(self.num_timesteps)

            if episode_num > 20000 and ave_return > self.best_ave_return:
                if self.verbose >= 1:
                    print("New best average return!")

                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))

                self.best_ave_return = float(ave_return)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


class CurriculumCallback(BaseCallback):
    def __init__(self, verbose: int = 0, use_sac: bool = False):
        super().__init__(verbose=verbose)
        self.use_sac: bool = use_sac

    def _on_step(self) -> bool:
        if self.locals['done']:
            episode_num = self.locals['self']._episode_num
            # print("\033[92m", "TRAIN in Episode: ", episode_num, ", Result: ", self.locals['info'], "\033[0m")
            # logging.info("\033[92m", "TRAIN in Episode: ", episode_num, ", Result: ", self.locals['info'], "\033[0m")
            logging.info(f"\033[92mTRAIN in Episode: {episode_num}, Result: {self.locals['info']}\033[0m")
            # if episode_num == 1999:
            #     self.locals['self'].sample_ratio = 0.5
            # elif episode_num == 5999:
            #     self.locals['self'].sample_ratio = 0.25
            # elif episode_num == 9999:
            #     return False
            # if episode_num == 9999:
            #     return False

            if self.use_sac:
                if episode_num == 29999:
                    return False
            else:
                # if episode_num == 3999:
                #     self.locals['env'].set_phase(1)
                #     logging.info(f"\033[92mCurriculum learning enters stage 2.\033[0m")
                # elif episode_num == 9999:
                #     self.locals['env'].set_phase(2)
                #     logging.info(f"\033[92mCurriculum learning enters stage 3.\033[0m")
                # elif episode_num == 15999:
                #     self.locals['env'].set_phase(3)
                #     logging.info(f"\033[92mCurriculum learning enters terminal stage.\033[0m")
                # elif episode_num == 29999:
                #     self.locals['env'].set_phase(5)
                #     logging.info(f"\033[92mCurriculum learning enters terminal stage.\033[0m")
                # elif episode_num == 39999:
                #     return False


                if episode_num == 3999:
                    self.locals['env'].set_phase(1)
                    logging.info(f"\033[92mCurriculum learning enters stage 2.\033[0m")
                elif episode_num == 11999:
                    self.locals['env'].set_phase(2)
                    logging.info(f"\033[92mCurriculum learning enters stage 3.\033[0m")
                elif episode_num == 19999:
                    self.locals['env'].set_phase(3)
                    logging.info(f"\033[92mCurriculum learning enters terminal stage.\033[0m")
                # elif episode_num == 39999:
                #     self.locals['env'].set_phase(5)
                #     logging.info(f"\033[92mCurriculum learning enters terminal stage.\033[0m")
                elif episode_num == 49999:
                    return False





        return True

class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
            self,
            save_freq: int,
            save_path: str,
            name_prefix: str = "rl_model",
            save_replay_buffer: bool = False,
            # save_vecnormalize: bool = False,
            verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        # self.save_vecnormalize = save_vecnormalize

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        episode_num = self.locals['self']._episode_num + 1
        return os.path.join(self.save_path,
                            f"{self.name_prefix}_{checkpoint_type}{episode_num}_steps.{extension}")

    def _on_step(self) -> bool:

        episode_num = self.locals['self']._episode_num

        if episode_num > 16000 and  episode_num % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model,
                                                   "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            # if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
            #     # Save the VecNormalize statistics
            #     vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
            #     self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
            #     if self.verbose >= 2:
            #         print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True
