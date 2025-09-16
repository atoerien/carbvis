from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    Iterable,
    ParamSpec,
    TypeVar,
)

import numpy as np
from numpy.typing import NDArray

TIMING = False
# TIMING = True

# stop complaining when assigning float32 to float
if TYPE_CHECKING:
    float = float | np.floating


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]
UByteArray = NDArray[np.uint8]

T = TypeVar("T")

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


def spline(
    a: FloatArray,
    b: FloatArray,
    c: FloatArray,
    d: FloatArray,
    t: float | FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Calculates the position at point(s) t along the spline
    with co-efficients A, B, C and D.
    spline(t) = ((A * t + B) * t + C) * t + D
    """
    t = np.asarray(t).reshape(-1, 1)

    shape = np.broadcast(t, d).shape

    path = np.copy(np.broadcast_to(d, shape))
    path += t * c
    path += t**2 * b
    path += t**3 * a

    tan = np.copy(np.broadcast_to(c, shape))
    tan += 2 * t * b
    tan += 3 * t**2 * a
    tan /= np.linalg.norm(tan, axis=1).reshape(-1, 1)

    return path, tan


def rotate(v: FloatArray, axis: FloatArray, angle: float) -> FloatArray:
    """Rotate vector(s) v about axis by angle using Rodrigues' formula.

    v: shape (3,) or (n, 3)
    axis: shape (3,)
    angle: float
    """
    axis = axis / np.linalg.norm(axis)
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)

    v = np.atleast_2d(v)  # ensure (n, 3) shape

    # dot product per row
    dot = np.dot(v, axis)[:, None]  # shape (n, 1)

    rotated = cos_a * v + (1 - cos_a) * dot * axis + sin_a * np.cross(axis, v)

    return rotated if v.shape[0] > 1 else rotated[0]


def xyz_to_spherical(xyz: FloatArray, axis=-1) -> FloatArray:
    x, y, z = np.moveaxis(xyz, axis, 0)

    lea = np.empty_like(xyz)

    pre_selector = ((slice(None),) * lea.ndim)[:axis]

    xy_sq = x**2 + y**2
    lea[(*pre_selector, 0)] = np.sqrt(xy_sq + z**2)
    lea[(*pre_selector, 1)] = np.arctan2(np.sqrt(xy_sq), z)
    lea[(*pre_selector, 2)] = np.arctan2(y, x)

    return lea


def spherical_to_xyz(lea: FloatArray, axis=-1) -> FloatArray:
    l, e, a = np.moveaxis(lea, axis, 0)

    xyz = np.empty_like(lea)

    pre_selector = ((slice(None),) * xyz.ndim)[:axis]

    xyz[(*pre_selector, 0)] = l * np.sin(e) * np.cos(a)
    xyz[(*pre_selector, 1)] = l * np.sin(e) * np.sin(a)
    xyz[(*pre_selector, 2)] = l * np.cos(e)

    return xyz


def gaussian(x, a, b, c, p=1):
    return a * np.exp(-((((x - b) ** 2) / (2 * (c**2))) ** p))


def dfs_paths(
    get_neighbors: Callable[[T], Iterable[T]],
    node: T,
    visited: set[int] | None = None,
    path: list[T] | None = None,
) -> Generator[list[T]]:
    if path is None:
        path = [node]
    if visited is None:
        visited = {id(node)}

    extended = False
    for neighbor in get_neighbors(node):
        id_neighbor = id(neighbor)
        if id_neighbor not in visited:
            path.append(neighbor)
            visited.add(id_neighbor)
            yield from dfs_paths(get_neighbors, neighbor, visited, path)
            visited.remove(id_neighbor)
            path.pop()
            extended = True
    if not extended:  # hit an endpoint, return the path
        yield path.copy()


def color_float_to_ubyte(arr: FloatArray) -> UByteArray:
    """
    Convert float colors in [0, 1] to uint8 in [0, 255].
    Supports shape (..., 3) or (..., 4).
    If RGB (last dim=3), appends alpha=255.
    """

    if arr.shape[-1] not in (3, 4):
        raise ValueError("Last dimension must be 3 (RGB) or 4 (RGBA)")

    # Avoid modifying caller's array
    arr = arr.copy()

    # Clamp to [0, 1] and scale to [0, 255]
    np.clip(arr, 0.0, 1.0, out=arr)
    arr *= 255.0

    # RGBA
    if arr.shape[-1] == 4:
        return arr.astype(np.uint8)

    # RGB, append 255
    ret = np.empty((*arr.shape[:-1], 4), dtype=np.uint8)
    ret[..., :3] = arr
    ret[..., 3] = 255
    return ret


@dataclass(slots=True)
class Frame:
    """
    Frame: A frame of reference at a point.
    A frame has an origin and 3 basis vectors, plus an approximate cumulative arc
    length (used for texturing).
    """

    origin: FloatArray
    forward: FloatArray
    right: FloatArray
    up: FloatArray
    arclength: float = 0.0

    def copy(self):
        return Frame(
            origin=np.copy(self.origin),
            forward=np.copy(self.forward),
            right=np.copy(self.right),
            up=np.copy(self.up),
            arclength=self.arclength,
        )

    def align(self, forward: FloatArray):
        """Align frame to a forward vector"""

        rot_axis = np.cross(self.forward, forward)
        axis_norm = np.linalg.norm(rot_axis)

        # skip rotation if we're already aligned
        if axis_norm > 1e-4:
            rot_angle = np.arccos(
                # float shenanigans
                np.clip(np.dot(self.forward, forward), -1.0, 1.0)
            )

            # rotate frame angle rot_angle about rot_axis
            self.forward = rotate(self.forward, rot_axis, rot_angle)
            self.right = rotate(self.right, rot_axis, rot_angle)
            self.up = rotate(self.up, rot_axis, rot_angle)
