import logging
import datetime
import sys
import time

_timer_log_level = 15

# setup logger
logging.addLevelName(_timer_log_level, "PERF")
logger = logging.getLogger("stgeotk")
formatter = logging.Formatter(
    "[%(levelname)s][%(asctime)s] %(name)s: %(message)s")


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __debug__:
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
else:
    logger.setLevel(logging.INFO)
    log_filename = "stgeotk_" + \
        datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


seconds_per_year = 60 * 60 * 24 * 365.2425


def log_info(msg):
    logger.info("%s", BColors.OKGREEN + msg + BColors.ENDC)


def current_function_name(level=1):
    return sys._getframe(level).f_code.co_name


def second_to_myr(seconds):
    return seconds / seconds_per_year / 1.0e6


def meter_per_second_to_cm_per_year(meter_per_second):
    return seconds_per_year * 100. * meter_per_second


class TimerError(Exception):
    """
    Custom exception for Timer.
    """


class Timer:
    """
    A simple implementation for a code timer.
    usage:
        timer = Timer()
        timer.start()
        elapsed_seconds = timer.stop()
    or using context manager:
        with Timer() as _:
            <...code to be timed...>
    """

    def __init__(self):
        self._start_time = None
        self._label_text = ""

    def __enter__(self):
        self._label_text = current_function_name(2)
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        '''
        Start new timer
        The function name will also be recorded for reporting
        '''
        if self._start_time is not None:
            raise TimerError(
                "Timer is running. Use stop() to stop the previous run.")
        self._label_text = current_function_name(2)
        self._start_time = time.perf_counter()

    def stop(self):
        '''
        Stop the timer and return the elapsed time.
        Also log the time to the global logger at loglevel "TIMER"
        '''
        if self._start_time is None:
            raise TimerError("Timer is not yet started")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        logger.log(_timer_log_level, "%s",
                   f"{BColors.OKBLUE}{BColors.BOLD}[{self._label_text}]"
                   f"{BColors.ENDC}{BColors.OKBLUE} elpased {elapsed_time} "
                   f"seconds{BColors.ENDC}")
        return elapsed_time
