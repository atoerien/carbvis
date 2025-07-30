import functools
from typing import Callable, ParamSpec, TypeVar

TIMING = False
TIMING = True

P = ParamSpec("P")
R = TypeVar("R")


def time(fn: Callable[P, R]) -> Callable[P, R]:
    if not TIMING:
        return fn

    from time import time

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time()
        result = fn(*args, **kwargs)
        end = time()
        duration = end - start
        print(f"{fn.__qualname__}(): {duration:.4f}")
        return result

    return wrapper
