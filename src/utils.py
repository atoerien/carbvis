from __future__ import annotations

from typing import Callable, Generator, Iterable, TypeVar

import numpy as np
from numpy.typing import NDArray

from ._utils import Frame as Frame  # pyright: ignore[reportMissingModuleSource]

FloatArray = NDArray[np.float32]
DoubleArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
UByteArray = NDArray[np.uint8]

T = TypeVar("T")


def gaussian(x, a, b, c, p=1):
    """super-Gaussian function of order p"""
    return a * np.exp(-((((x - b) ** 2) / (2 * (c**2))) ** p))


def _dfs_paths_id(node) -> int:
    if type(node) is tuple:
        return id(node[0])
    else:
        return id(node)


def dfs_paths(
    get_neighbors: Callable[[T], Iterable[T]],
    node: T,
    visited: set[int] | None = None,
    path: list[T] | None = None,
    *,
    max_len: int = 0,
) -> Generator[list[T]]:
    """
    Depth-first search (DFS) path generator for arbitrary graphs.

    This function explores all simple acyclic paths starting from a
    given node using depth-first search. It yields each path as a list
    of nodes from the start node to a leaf (endpoint) where no further
    unvisited neighbors exist.

    Node identity is tracked using `id()`, rather than using == (`__eq__()`).
    If the nodes are tuples, then `id(node[0])` is used, otherwise `id(node)`.
    This is intended to be used for auxiliary data about either nodes or edges.

    Args:
        get_neighbors: A function that returns the neighbors of a node.
        node: The starting node for the DFS.
        visited: A set of visited node IDs used to avoid revisiting nodes.
            If None, a new set is created.
        path: The current path of nodes being explored. If None, a new
            path is started with the given node.
        max_len: The maximum path length. If 0, no limit.

    Yields:
        A list of nodes representing one complete path, from the
        starting node to a terminal node.
    """

    if path is None:
        path = [node]
    if visited is None:
        visited = {_dfs_paths_id(node)}

    extended = False
    for neighbor in get_neighbors(node):
        id_neighbor = _dfs_paths_id(neighbor)
        if id_neighbor not in visited:
            # we're at the max len and we have a neighbor,
            # this path is invalid and we can return early
            if max_len == 1:
                return

            path.append(neighbor)
            visited.add(id_neighbor)
            yield from dfs_paths(
                get_neighbors,
                neighbor,
                visited,
                path,
                max_len=max_len - 1,
            )
            visited.remove(id_neighbor)
            path.pop()
            extended = True
    if not extended:  # hit an endpoint, return the path
        yield path.copy()


def color_float_to_ubyte(arr: FloatArray) -> UByteArray:
    """
    Convert floating-point RGBA colors to unsigned integer format.

    Values are clamped to the range [0, 1], scaled to [0, 255], and returned
    as uint8.

    Args:
        arr: A array of floats representing colors in the range [0, 1].
            The last dimension must be length 4 (RGBA).

    Returns:
        An array of dtype uint8 with the same shape as `arr`.
    """

    if arr.shape[-1] != 4:
        raise ValueError("Last dimension must be length 4 (RGBA)")

    # avoid modifying caller's array
    arr = arr.copy()

    # clamp to [0, 1] and scale to [0, 255]
    np.clip(arr, 0.0, 1.0, out=arr)
    arr *= 255.0

    return arr.astype(np.uint8)
