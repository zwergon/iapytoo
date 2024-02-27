# timer.py

import time
import numpy as np
import logging


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    """Time your code using a class, context manager, or decorator"""

    text: str = "{} cycle elapsed {:0.4f} s mean {:0.4f}"
    with_log: bool = True

    def __init__(self):
        self._start_time = None
        self.times = []

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        self._start_time = None

    def tick(self) -> None:
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")
        self.times.append(time.perf_counter() - self._start_time)
        self._start_time = time.perf_counter()

    @property
    def elapsed(self):
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")
        if len(self.times) == 0:
            self.tick()
        return np.sum(self.times)

    def log(self):
        if self.with_log:
            elapsed_time = self.elapsed
            logging.info(
                self.text.format(len(self.times), elapsed_time, np.mean(self.times))
            )
