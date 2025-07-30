from __future__ import annotations

import numpy as np
from chimerax.atomic import Structure
from chimerax.graphics import Drawing

from .carbs import (
    find_small_rings,
    get_ring_centroid_and_normal,
)
from .utils import time


def get_ring_color(ring_coords: np.ndarray):
    return hill_reilly_ring_color(ring_coords)


def hill_reilly_ring_color(ring_coords: np.ndarray):
    pucker_sum = hill_reilly_ring_pucker(ring_coords)
    return hotcold_gradient(pucker_sum)


def hill_reilly_ring_pucker(ring_coords: np.ndarray) -> float:
    displ = atom_displ_from_mean_plane(ring_coords)

    # XXX: isn't this just cremer-pople?
    q = np.sqrt(np.sum(displ**2))
    return min(q, 2.0)  # truncate amplitude at 2


def atom_displ_from_mean_plane(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]

    # calculate centre of geometry
    cog = np.mean(coords, axis=0)

    # centre the ring
    centered = coords - cog

    # Calculate cartesian axes based on coords of nuclei in ring
    # using cremer-pople algorithm. It is assumed that the
    # centre of geometry is the centre of the ring.

    indices = np.arange(n)
    ze_angle = 2.0 * np.pi * (indices - 1) / n
    ze_sin = np.sin(ze_angle)
    ze_cos = np.cos(ze_angle)

    Rp = np.sum(centered.T * ze_sin, axis=1)
    Rpp = np.sum(centered.T * ze_cos, axis=1)

    z = np.cross(Rp, Rpp)
    z /= np.linalg.norm(z)

    # lmbda = np.dot(z, Rp)
    # y = Rp - z * lmbda
    # y /= np.linalg.norm(y)

    # x = np.cross(y, z)

    # calculate displacement from mean plane
    return np.dot(centered, z)


def hotcold_gradient(pucker_sum: float):
    rgb = np.zeros(3, dtype=np.float32)  # default color is black

    # Hot to cold color map:
    # Red -> Yellow -> Green -> Cyan -> Blue -> Magenta
    if pucker_sum < 0.40:
        # Red (1,0,0) -> Yellow (1,1,0)
        rgb[0] = 1.0  # red
        rgb[1] = pucker_sum * 2.5  # increase green -> yellow
        rgb[2] = 0.0
    elif pucker_sum < 0.56:
        # Yellow (1,1,0) -> Green (0,1,0)
        rgb[0] = 1.0 - (pucker_sum - 0.40) * 6.25  # decrease red -> green
        rgb[1] = 1.0
        rgb[2] = 0.0
    elif pucker_sum < 0.64:
        # Green (0,1,0) -> Cyan (0,1,1)
        rgb[0] = 0.0
        rgb[1] = 1.0  # green
        rgb[2] = (pucker_sum - 0.56) * 12.5  # increase blue
    elif pucker_sum < 0.76:
        # Cyan (0,1,1) -> Blue (0,0,1)
        rgb[0] = 0.0
        rgb[1] = 1.0 - (pucker_sum - 0.64) * 5.0  # decrease green
        rgb[2] = 1.0
    else:
        # Blue (0,0,1) -> Magenta (1,0,1)
        rgb[0] = (pucker_sum - 0.76) * 0.8  # increase red
        rgb[1] = 0.0
        rgb[2] = 1.0

    # Clamp to [0, 1] and scale to [0, 255]
    rgb = np.clip(rgb, 0.0, 1.0) * 255.0

    # Round and convert to uint8, then append alpha=255
    rgba = np.empty(4, dtype=np.uint8)
    rgba[:3] = np.round(rgb).astype(np.uint8)
    rgba[3] = 255  # alpha channel

    return rgba


def _make_paperchain_graphics(
    structure: Structure,
    paperchain_drawing: PaperChainDrawing,
):
    paperchain_drawing.clear()

    # FIXME: not hardcode
    maxringsize = 10
    bipyramid_height = 1.0

    rings = find_small_rings(structure.atoms, maxringsize)

    vertices = []
    normals = []
    colors = []
    triangles = []

    for ring in rings:
        n = len(ring.atoms)  # the number of atoms in the current ring

        ring_coords = ring.get_ring_coords()

        rgba = get_ring_color(ring_coords)
        centroid, normal = get_ring_centroid_and_normal(ring_coords)

        x = 0.5 * bipyramid_height * normal
        top = centroid + x
        bottom = centroid - x

        triangle_offset = len(vertices)

        vertices.append(top)
        normals.append(normal)
        colors.append(rgba)

        vertices.append(bottom)
        normals.append(normal)
        colors.append(rgba)

        # draw top half
        for i in range(n):
            # calculate next ring position (wrapping as necessary)
            next_i = i + 1
            if next_i >= n:
                next_i = 0

            curvec = ring_coords[i]
            nextvec = ring_coords[next_i]

            tmp0 = curvec - nextvec
            tmp1 = nextvec - top
            normtop = np.cross(tmp0, tmp1)
            normtop /= np.linalg.norm(normtop)

            vertices.append(curvec)
            normals.append(normtop)
            colors.append(rgba)

            triangles.append(
                (
                    triangle_offset + 2 + i,  # curvec
                    triangle_offset + 2 + next_i,  # nextvec
                    triangle_offset,  # top
                )
            )

        # draw bottom half
        for i in range(n):
            # calculate next ring position (wrapping as necessary)
            next_i = i + 1
            if next_i >= n:
                next_i = 0

            curvec = ring_coords[i]
            nextvec = ring_coords[next_i]

            tmp0 = curvec - nextvec
            tmp1 = nextvec - bottom
            normbot = np.cross(tmp0, tmp1)
            normbot /= np.linalg.norm(normbot)

            vertices.append(curvec)
            normals.append(normbot)
            colors.append(rgba)

            triangles.append(
                (
                    triangle_offset + 2 + n + i,  # curvec
                    triangle_offset + 2 + n + next_i,  # nextvec
                    triangle_offset + 1,  # bottom
                )
            )

    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    triangles = np.array(triangles, dtype=np.int32)

    # paperchain_drawing.display_style = Drawing.Mesh
    paperchain_drawing.set_geometry(vertices, normals, triangles)
    paperchain_drawing.set_vertex_colors(colors)


class PaperChainDrawing(Drawing):
    def __init__(self, name="paperchain"):
        super().__init__(name)

    def clear(self):
        self.set_geometry(None, None, None)
        self.remove_all_drawings()

    @time
    def compute_paperchain(self, structure: Structure):
        _make_paperchain_graphics(structure, self)
