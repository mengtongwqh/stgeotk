import logging
import datetime
import time

_timer_log_level = 15

# setup logger
logging.addLevelName(_timer_log_level, "TIMER")
logger = logging.getLogger("stgeotk")
formatter = logging.Formatter(
    "[%(levelname)s][%(asctime)s] %(name)s: %(message)s")


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


def current_function_name(level=1):
    import sys
    return sys._getframe(level).f_code.co_name


def second_to_myr(seconds):
    return seconds / seconds_per_year / 1.0e6


def meter_per_second_to_cm_per_year(meter_per_second):
    return seconds_per_year * 100. * meter_per_second


def deref_or_default(kv, key, default):
    if key in kv:
        return kv[key]
    else:
        return default


class TimerError(Exception):
    """
    Custom exception for Timer.
    """


class Timer:
    '''
    A simple implementation for a code timer.
    usage:
    timer = Timer()
    timer.start()
    elapsed_seconds = timer.stop()
    '''

    def __init__(self):
        self._start_time = None

    def start(self):
        '''
        Start new timer
        The function name will also be recorded for reporting
        '''
        if self._start_time is not None:
            raise TimerError(
                "Timer is running. Use stop() to stop the previous run.")
        self._start_time = time.perf_counter()
        self._label_text = current_function_name(2)

    def stop(self):
        '''
        Stop the timer and return the elapsed time.
        Also log the time to the global logger at loglevel "TIMER"
        '''
        if self._start_time is None:
            raise TimerError("Timer is not yet started")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        logger.log(_timer_log_level, "Timer in [{0}] elapsed {1} seconds".\
            format(self._label_text, elapsed_time))
        return elapsed_time
