import logging
import tracemalloc
from functools import wraps


def memory_stats(func):

    @wraps
    def wrapper(*args, **kwargs):
        tracemalloc.start()

        res = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Current memory usage: {current / 10**6:.2f} MB")
        logging.info(f"Peak memory usage:    {peak / 10**6:.2f} MB")

        return res

    return wrapper
