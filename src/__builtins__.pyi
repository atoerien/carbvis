from typing import Callable, TypeVar

T = TypeVar("T", bound=Callable)

def line_profile(fn: T) -> T: ...
