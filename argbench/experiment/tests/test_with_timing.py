from unittest import *
import time
import logging

logger = logging.getLogger(__name__)

def with_timing(fn):
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            e= time.perf_counter()
            print(f"Time for {fn} is {e-t:2.2f}")
            logger.log (level=logging.INFO, msg=f"Time for {fn} is {e-t:2.2f}")
    return wrapper

@with_timing
def fun():
    for i in range(100000):
        pass

class TestTiming:
    def test_timing(self):
        fun()