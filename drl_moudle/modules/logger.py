import datetime
import os
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import matplotlib.figure
import numpy as np
import pandas
import torch as th

try:
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.tensorboard.summary import hparams
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50


class HParam:
    """
    Hyperparameter data class storing hyperparameters and metrics in dictionaries

    :param hparam_dict: key-value pairs of hyperparameters to log
    :param metric_dict: key-value pairs of metrics to log
        A non-empty metrics dict is required to display hyperparameters in the corresponding Tensorboard section.
    """

    def __init__(self, hparam_dict: Mapping[str, Union[bool, str, float, None]], metric_dict: Mapping[str, float]):
        self.hparam_dict = hparam_dict
        if not metric_dict:
            raise Exception("`metric_dict` must not be empty to display hyperparameters to the HPARAMS tensorboard tab.")
        self.metric_dict = metric_dict


class KVWriter:
    """
    Key Value writer
    """

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Tuple[str, ...]], step: int = 0) -> None:
        """
        Write a dictionary to file

        :param key_values:
        :param key_excluded:
        :param step:
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close owned resources
        """
        raise NotImplementedError


class SeqWriter:
    """
    sequence writer
    """

    def write_sequence(self, sequence: List[str]) -> None:
        """
        write_sequence an array to file

        :param sequence:
        """
        raise NotImplementedError


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.

    :param folder: the folder to write the log to
    """

    def __init__(self, folder: str):
        assert SummaryWriter is not None, "tensorboard is not installed, you can use `pip install tensorboard` to do so"
        self.writer = SummaryWriter(log_dir=folder)
        self._is_closed = False

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Tuple[str, ...]], step: int = 0) -> None:
        assert not self._is_closed, "The SummaryWriter was closed, please re-create one."
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            if excluded is not None and "tensorboard" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if isinstance(value, str):
                    # str is considered a np.ScalarType
                    self.writer.add_text(key, value, step)
                else:
                    self.writer.add_scalar(key, value, step)

            if isinstance(value, th.Tensor):
                self.writer.add_histogram(key, value, step)

            if isinstance(value, HParam):
                # we don't use `self.writer.add_hparams` to have control over the log_dir
                experiment, session_start_info, session_end_info = hparams(value.hparam_dict, metric_dict=value.metric_dict)
                self.writer.file_writer.add_summary(experiment)
                self.writer.file_writer.add_summary(session_start_info)
                self.writer.file_writer.add_summary(session_end_info)

        # Flush the output to the file
        self.writer.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.writer:
            self.writer.close()
            self._is_closed = True


def make_output_format(_format: str, log_dir: str, log_suffix: str = "") -> KVWriter:
    """
    return a logger for the requested format

    :param _format: the requested format to log to ('stdout', 'log', 'json' or 'csv' or 'tensorboard')
    :param log_dir: the logging directory
    :param log_suffix: the suffix for the log file
    :return: the logger
    """
    os.makedirs(log_dir, exist_ok=True)
    if _format == "stdout":
        pass
    elif _format == "tensorboard":
        return TensorBoardOutputFormat(log_dir)
    else:
        raise ValueError(f"Unknown format specified: {_format}")


# ================================================================
# Backend
# ================================================================


class Logger:
    """
    The logger class.

    :param folder: the logging location
    :param output_formats: the list of output formats
    """

    def __init__(self, folder: Optional[str], output_formats: List[KVWriter]):
        self.name_to_value: Dict[str, float] = defaultdict(float)  # values this iteration
        self.name_to_count: Dict[str, int] = defaultdict(int)
        self.name_to_excluded: Dict[str, Tuple[str, ...]] = {}
        self.level = INFO
        self.dir = folder
        self.output_formats = output_formats

    @staticmethod
    def to_tuple(string_or_tuple: Optional[Union[str, Tuple[str, ...]]]) -> Tuple[str, ...]:
        """
        Helper function to convert str to tuple of str.
        """
        if string_or_tuple is None:
            return ("",)
        if isinstance(string_or_tuple, tuple):
            return string_or_tuple
        return (string_or_tuple,)

    def record(self, key: str, value: Any, exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: save to log this key
        :param value: save to log this value
        :param exclude: outputs to be excluded
        """
        self.name_to_value[key] = value
        self.name_to_excluded[key] = self.to_tuple(exclude)

    def record_mean(self, key: str, value: Optional[float], exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        """
        The same as record(), but if called many times, values averaged.

        :param key: save to log this key
        :param value: save to log this value
        :param exclude: outputs to be excluded
        """
        if value is None:
            return
        old_val, count = self.name_to_value[key], self.name_to_count[key]
        self.name_to_value[key] = old_val * count / (count + 1) + value / (count + 1)
        self.name_to_count[key] = count + 1
        self.name_to_excluded[key] = self.to_tuple(exclude)

    def dump(self, step: int = 0) -> None:
        """
        Write all of the diagnostics from the current iteration
        """
        if self.level == DISABLED:
            return
        for _format in self.output_formats:
            if isinstance(_format, KVWriter):
                _format.write(self.name_to_value, self.name_to_excluded, step)

        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()

    def log(self, *args, level: int = INFO) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).

        level: int. (see logger.py docs) If the global logger level is higher than
                    the level argument here, don't print to stdout.

        :param args: log the arguments
        :param level: the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        """
        if self.level <= level:
            self._do_log(args)

    def debug(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the DEBUG level.

        :param args: log the arguments
        """
        self.log(*args, level=DEBUG)

    def info(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the INFO level.

        :param args: log the arguments
        """
        self.log(*args, level=INFO)

    def warn(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the WARN level.

        :param args: log the arguments
        """
        self.log(*args, level=WARN)

    def error(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the ERROR level.

        :param args: log the arguments
        """
        self.log(*args, level=ERROR)

    # Configuration
    # ----------------------------------------
    def set_level(self, level: int) -> None:
        """
        Set logging threshold on current logger.

        :param level: the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        """
        self.level = level

    def get_dir(self) -> Optional[str]:
        """
        Get directory that log files are being written to.
        will be None if there is no output directory (i.e., if you didn't call start)

        :return: the logging directory
        """
        return self.dir

    def close(self) -> None:
        """
        closes the file
        """
        for _format in self.output_formats:
            _format.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args: Tuple[Any, ...]) -> None:
        """
        log to the requested format outputs

        :param args: the arguments to log
        """
        for _format in self.output_formats:
            if isinstance(_format, SeqWriter):
                _format.write_sequence(list(map(str, args)))


def configure(folder: Optional[str] = None, format_strings: Optional[List[str]] = None) -> Logger:
    """
    Configure the current logger.

    :param folder: the save location
        (if None, $SB3_LOGDIR, if still None, tempdir/SB3-[date & time])
    :param format_strings: the output logging format
        (if None, $SB3_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    :return: The logger object.
    """
    if folder is None:
        folder = os.getenv("SB3_LOGDIR")
    if folder is None:
        folder = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S-%f"))

    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    log_suffix = ""
    if format_strings is None:
        format_strings = os.getenv("SB3_LOG_FORMAT", "stdout,log,csv").split(",")

    format_strings = list(filter(None, format_strings))
    output_formats = [make_output_format(f, folder, log_suffix) for f in format_strings]

    logger = Logger(folder=folder, output_formats=output_formats)
    # Only print when some files will be saved
    if len(format_strings) > 0 and format_strings != ["stdout"]:
        logger.log(f"Logging to {folder}")
    return logger
