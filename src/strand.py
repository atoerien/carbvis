from __future__ import annotations

import numpy as np
from chimerax.atomic import Structure
from chimerax.atomic.changes import Changes
from chimerax.core.session import Session
from chimerax.geometry import natural_cubic_spline

from .carbs import (
    CarbChain,
    dihedral_norm_colormap,
    find_chains,
    find_linkages,
    find_rings,
)
from .model import CarbVisModel
from .utils import FloatArray, Frame, color_float_to_ubyte, rotate, time


def draw_tube(
    vertices: list[FloatArray],
    normals: list[FloatArray],
    triangles: list[tuple[int, int, int]],
    vcolors: list[FloatArray],
    chain: CarbChain,
    radius: float = 1.0,
    segment_subdivisions: int = 20,
    circle_subdivisions: int = 16,
    candy_cane: bool = False,
):
    linkages = chain.linkages
    if not linkages:
        return

    color_b = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    points = np.empty((3 * len(linkages) + 1, 3), dtype=np.float32)
    norms = np.empty((3 * len(linkages) + 1, 3), dtype=np.float32)
    colors = np.empty((3 * len(linkages) + 1, 3), dtype=np.float32)
    for i, link in enumerate(linkages):
        start_centroid, start_normal = link.start_ring.get_centroid_and_normal()
        end_centroid, end_normal = link.end_ring.get_centroid_and_normal()

        angles = link.calc_angles()
        color = dihedral_norm_colormap(angles)

        j = 3 * i
        points[j] = start_centroid
        norms[j] = start_normal
        colors[j] = (colors[j - 1] + color) / 2 if i != 0 else color
        points[j + 1] = link.atoms[0].coord
        norms[j + 1] = start_normal
        colors[j + 1] = color
        points[j + 2] = link.atoms[-1].coord
        norms[j + 2] = end_normal
        colors[j + 2] = color
        if i == len(linkages) - 1:
            points[j + 3] = end_centroid
            norms[j + 3] = end_normal
            colors[j + 3] = color

    # TODO: add fake point for carbon 1 on start and end
    # to make tangent at centre better so side chains connect more convincingly
    # TODO: also maybe don't use natural_cubic_spline and use our own tangents

    spath, stan = natural_cubic_spline(points, segment_subdivisions)

    scolors = natural_cubic_spline(colors, segment_subdivisions, tangents=False)

    frames: list[Frame] = []

    # Initial frame
    point = spath[0]
    tangent = stan[0]
    up = norms[0]
    up = up - np.dot(up, tangent) * tangent
    if np.allclose(up, 0.0):
        # fallback: pick a world axis
        if abs(tangent[0]) < 0.9:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up = up - np.dot(up, tangent) * tangent
    up /= np.linalg.norm(up)

    # tangent and forward are length 1, no need to norm
    right = np.cross(tangent, up)

    frames.append(Frame(point, tangent, right, up))

    # Threshold to determine when we have a reliable
    # rotation axis to rotate a frame about
    min_axis_norm = 1e-4

    for i in range(1, len(spath)):
        prev_frame = frames[-1]
        point = spath[i]
        tangent = stan[i]

        rot_axis = np.cross(prev_frame.forward, tangent)
        axis_norm = np.linalg.norm(rot_axis)

        # copy previous frame
        frame = prev_frame.copy()
        frame.origin = point
        frame.arclength += np.linalg.norm(point - prev_frame.origin)

        # rotate frame if tangents not parallel
        if axis_norm > min_axis_norm:
            rot_angle = np.arccos(
                # float shenanigans
                np.clip(np.dot(prev_frame.forward, tangent), -1.0, 1.0)
            )

            # rotate frame angle rot_angle about rot_axis
            frame.forward = rotate(frame.forward, rot_axis, rot_angle)
            frame.right = rotate(frame.right, rot_axis, rot_angle)
            frame.up = rotate(frame.up, rot_axis, rot_angle)

        frames.append(frame)

    if candy_cane:
        # TODO: for single-ring side chains, ~180º flips often
        # twist the other way as twister

        twist = np.empty((3 * len(linkages) + 1, 1), dtype=np.float32)

        for i, norm in enumerate(norms):
            iframe = i * (segment_subdivisions + 1)
            frame = frames[iframe]

            t = np.arccos(
                # float shenanigans
                np.clip(np.dot(norm, frame.up), -1.0, 1.0)
            )
            if np.dot(np.cross(norm, frame.up), frame.forward) > 0:
                t *= -1

            if i != 0:
                t_prev = twist[i - 1]
                d = (t - t_prev + np.pi) // (2 * np.pi)
                if d != 0:
                    t -= d * 2 * np.pi

            twist[i] = t

        stwist = natural_cubic_spline(twist, segment_subdivisions, tangents=False)
    else:
        stwist = None

    triangle_offset = len(vertices)

    for i, frame in enumerate(frames):
        if candy_cane:
            # rotate frame about frame.forward (twist)
            assert stwist is not None
            t = stwist[i]
            frame.right = rotate(frame.right, frame.forward, t)
            frame.up = rotate(frame.up, frame.forward, t)

        a = np.linspace(0, 2 * np.pi, circle_subdivisions, endpoint=False)
        n = np.cos(a).reshape(-1, 1) * frame.up - np.sin(a).reshape(-1, 1) * frame.right
        v = frame.origin + radius * n

        color_a = scolors[i]

        if i == 0:
            # end cap vertices
            vertices.extend(v)
            normals.extend(-frame.forward for _ in range(circle_subdivisions))
            vcolors.extend(color_a for _ in range(circle_subdivisions))

        # tube vertices

        vertices.extend(v)
        normals.extend(n)
        vcolors.extend(
            # (color_a if j > 1 else np.array([0.0, 0.0, 0.0], dtype=np.float32))
            # for j in range(circle_subdivisions)
            color_a
            for _ in range(circle_subdivisions)
        )

        if candy_cane:
            vertices.extend(v)
            normals.extend(n)
            vcolors.extend(color_b for _ in range(circle_subdivisions))

        if i == len(frames) - 1:
            # end cap vertices
            vertices.extend(v)
            normals.extend(frame.forward for _ in range(circle_subdivisions))
            vcolors.extend(color_a for _ in range(circle_subdivisions))

            continue  # no normal triangles for the last frame

        if candy_cane:
            offset = triangle_offset + (2 * i + 1) * circle_subdivisions
            offset_next = triangle_offset + (2 * (i + 1) + 1) * circle_subdivisions
            for j in range(0, circle_subdivisions, 2):
                k0 = offset + j
                k1 = offset + (j + 1) % circle_subdivisions
                k2 = offset_next + j
                k3 = offset_next + (j + 1) % circle_subdivisions
                triangles.append((k0, k2, k1))
                triangles.append((k1, k2, k3))

            offset = triangle_offset + (2 * i + 2) * circle_subdivisions
            offset_next = triangle_offset + (2 * (i + 1) + 2) * circle_subdivisions
            for j in range(1, circle_subdivisions, 2):
                k0 = offset + j
                k1 = offset + (j + 1) % circle_subdivisions
                k2 = offset_next + j
                k3 = offset_next + (j + 1) % circle_subdivisions
                triangles.append((k0, k2, k1))
                triangles.append((k1, k2, k3))
        else:
            offset = triangle_offset + (i + 1) * circle_subdivisions
            offset_next = triangle_offset + (i + 2) * circle_subdivisions
            for j in range(circle_subdivisions):
                k0 = offset + j
                k1 = offset + (j + 1) % circle_subdivisions
                k2 = offset_next + j
                k3 = offset_next + (j + 1) % circle_subdivisions
                triangles.append((k0, k2, k1))
                triangles.append((k1, k2, k3))

    # end cap triangles

    offset = triangle_offset
    offset_next = triangle_offset + circle_subdivisions
    for j in range(0, circle_subdivisions // 2):
        k0 = offset + j
        k1 = offset + (j + 1)
        k2 = offset_next - (j + 1)
        k3 = k0 if j == 0 else offset_next - j
        triangles.append((k0, k1, k2))
        triangles.append((k2, k3, k0))

    offset = len(vertices) - circle_subdivisions
    offset_next = len(vertices)
    for j in range(0, circle_subdivisions // 2):
        k0 = offset + j
        k1 = offset + (j + 1)
        k2 = offset_next - (j + 1)
        k3 = k0 if j == 0 else offset_next - j
        triangles.append((k0, k2, k1))
        triangles.append((k3, k2, k0))


class StrandModel(CarbVisModel):
    def __init__(
        self,
        session: Session,
        structure: Structure,
        name: str | None = None,
        update=True,
    ):
        if name is None:
            name = f"{structure.name} Strand"
        super().__init__(session, structure, name, update)

    def _do_auto_update(self, changes: Changes):
        super()._do_auto_update(changes)

        # TODO: does this work for new bonds/atoms?
        if self._coordinates_changed(changes):
            self._recalculate_graphics()

    @time
    def _calc_graphics(self):
        # FIXME: not hardcode
        maxringsize = 10
        radius = 0.25

        rings = find_rings(self.structure, maxringsize)
        linkages = find_linkages(rings)
        chains = find_chains(linkages)

        vertices = []
        normals = []
        triangles = []
        vcolors = []

        for chain in chains:
            draw_tube(
                vertices,
                normals,
                triangles,
                vcolors,
                chain,
                radius=radius,
                candy_cane=False,
            )

        va = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        na = np.array(normals, dtype=np.float32).reshape(-1, 3)
        ta = np.array(triangles, dtype=np.int32).reshape(-1, 3)
        ca = np.array(vcolors, dtype=np.float32).reshape(-1, 3)

        self.set_geometry(va, na, ta)
        self.set_vertex_colors(color_float_to_ubyte(ca))
