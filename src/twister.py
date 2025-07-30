from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from chimerax.atomic import Structure
from chimerax.graphics import Drawing

from .carbs import (
    find_small_ring_linkages,
    find_small_rings,
    get_ring_centroid_and_normal,
)
from .utils import time

# stop complaining when assigning float32 to float
if TYPE_CHECKING:
    float = float | np.floating


@dataclass
class RibbonFrame:
    """
    Ribbon Frame: A frame of reference at a point along a ribbon being drawn by
                  using Twister algorithm.
    A frame has an origin and 3 basis vectors, plus and approximate cumulative arc
    length (used for texturing).
    """

    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    origin: np.ndarray
    arclength: float

    def copy(self):
        return RibbonFrame(
            forward=np.copy(self.forward),
            right=np.copy(self.right),
            up=np.copy(self.up),
            origin=np.copy(self.origin),
            arclength=self.arclength,
        )


def ribbon_spline(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, t: float):
    """
    Calculates the position at point t along the spline with co-efficients
    A, B, C and D.
    spline(t) = ((A * t + B) * t + C) * t + D
    """

    ret = np.copy(d)
    ret += t * c
    ret += t**2 * b
    ret += t**3 * a
    return ret


def rotate(v: np.ndarray, axis: np.ndarray, angle: float):
    """Rotate v about axis by angle using Rodrigues' formula."""

    axis = axis / np.linalg.norm(axis)
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    return cos_a * v + (1 - cos_a) * np.dot(axis, v) * axis + sin_a * np.cross(axis, v)


def twister_draw_ribbon_extensions(
    vertices: list[np.ndarray],
    colors: list[np.ndarray],
    normals: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    centroid: np.ndarray,
    normal: np.ndarray,
    right: np.ndarray,
    rib_point: np.ndarray,
    rib_height: float,
    rib_width: float,
    top_color: np.ndarray,
    bottom_color: np.ndarray,
):
    # TODO: fix lighting weirdness when not using soft lighting
    # (normals are wrong along the edges, need more vertices)

    triangle_offset = len(vertices)

    # vertices
    for height, color, norm in zip(
        (rib_height, -rib_height),
        (top_color, bottom_color),
        (normal, -normal),
    ):
        for point in (centroid, rib_point):
            for width in (rib_width, -rib_width):
                vert = point + height * normal + width * right
                vertices.append(vert)
                colors.append(color)
                normals.append(norm)

    # right
    triangles.append(
        (
            triangle_offset + 0,
            triangle_offset + 6,
            triangle_offset + 4,
        )
    )

    triangles.append(
        (
            triangle_offset + 0,
            triangle_offset + 2,
            triangle_offset + 6,
        )
    )

    # top
    triangles.append(
        (
            triangle_offset + 3,
            triangle_offset + 2,
            triangle_offset + 0,
        )
    )

    triangles.append(
        (
            triangle_offset + 3,
            triangle_offset + 0,
            triangle_offset + 1,
        )
    )

    # left
    triangles.append(
        (
            triangle_offset + 1,
            triangle_offset + 5,
            triangle_offset + 7,
        )
    )
    triangles.append(
        (
            triangle_offset + 1,
            triangle_offset + 7,
            triangle_offset + 3,
        )
    )

    # bottom
    triangles.append(
        (
            triangle_offset + 7,
            triangle_offset + 5,
            triangle_offset + 4,
        )
    )
    triangles.append(
        (
            triangle_offset + 7,
            triangle_offset + 4,
            triangle_offset + 6,
        )
    )


def twister_draw_hexagon(
    vertices: list[np.ndarray],
    colors: list[np.ndarray],
    normals: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    centroid: np.ndarray,
    normal: np.ndarray,
    start_coord: np.ndarray,
    rib_height: float,
    rib_width: float,
    top_color: np.ndarray,
    bottom_color: np.ndarray,
):
    # TODO: fix lighting weirdness when not using soft lighting
    # (normals are wrong along the edges, need more vertices)

    current_vertex = len(vertices)

    # top centroid vertex
    vert = centroid + rib_height * normal
    vertices.append(vert)
    colors.append(top_color)
    normals.append(normal)
    top_centroid_offset = current_vertex
    current_vertex += 1

    # bottom centroid vertex
    vert = centroid - rib_height * normal
    vertices.append(vert)
    colors.append(bottom_color)
    normals.append(-normal)
    bottom_centroid_offset = current_vertex
    current_vertex += 1

    # vertices for hexagon edges

    POLYGON_N = 12
    ROT_ANGLE = 2 * np.pi / POLYGON_N
    first_edge_offset = current_vertex

    current_vec = start_coord - centroid
    current_vec -= np.dot(current_vec, normal) * normal
    current_vec /= np.linalg.norm(current_vec)
    current_vec *= (1 / np.cos(ROT_ANGLE / 2)) * rib_width

    for i in range(POLYGON_N):
        polygon_point = centroid + current_vec

        # top hexagon vertex
        vert = polygon_point + rib_height * normal
        vertices.append(vert)
        colors.append(top_color)
        normals.append(normal)
        current_vertex += 1

        # bottom hexagon vertex
        vert = polygon_point - rib_height * normal
        vertices.append(vert)
        colors.append(bottom_color)
        normals.append(-normal)
        current_vertex += 1

        if i == POLYGON_N - 1:
            break  # don't bother rotating the last time

        # Rotate current vec pi/3 radians about the ring normal
        current_vec = rotate(current_vec, normal, ROT_ANGLE)

    for i in range(POLYGON_N):
        edge1_top = first_edge_offset + i * 2
        edge1_bottom = edge1_top + 1
        if i < POLYGON_N - 1:
            edge2_top = edge1_top + 2
            edge2_bottom = edge1_top + 3
        else:
            edge2_top = first_edge_offset
            edge2_bottom = first_edge_offset + 1

        # top
        triangles.append(
            (
                top_centroid_offset,
                edge1_top,
                edge2_top,
            )
        )

        # bottom
        triangles.append(
            (
                bottom_centroid_offset,
                edge2_bottom,
                edge1_bottom,
            )
        )

        # outer edge
        triangles.append(
            (
                edge1_bottom,
                edge2_bottom,
                edge1_top,
            )
        )
        triangles.append(
            (
                edge1_top,
                edge2_bottom,
                edge2_top,
            )
        )


def _make_twister_graphics(
    structure: Structure,
    twister_drawing: TwisterDrawing,
):
    twister_drawing.clear()

    # FIXME: not hardcode
    start_end_centroid = True
    hide_shared_links = False
    rib_steps = 10
    maxringsize = 10
    rib_width = 0.3
    rib_height = 0.05

    rings = find_small_rings(structure.atoms, maxringsize)
    linkages = find_small_ring_linkages(rings)
    print(f"Linkages: {len(linkages.links)} links, {len(linkages.paths)} paths")

    # Threshold to determine when we have a reliable
    # rotation axis to rotate a frame about
    min_axis_norm = 1e-4

    vertices = []
    normals = []
    colors = []
    triangles = []

    for path in linkages.paths:
        if hide_shared_links and linkages.shares_linkage_edges(path):
            continue

        start_ring = rings[path.start_ring]
        end_ring = rings[path.end_ring]

        start_ring_coords = start_ring.get_ring_coords()
        start_centroid, start_normal = get_ring_centroid_and_normal(start_ring_coords)

        end_ring_coords = end_ring.get_ring_coords()
        end_centroid, end_normal = get_ring_centroid_and_normal(end_ring_coords)

        start_coord = path.atoms[0].coord
        end_coord = path.atoms[-1].coord

        start_tangent = start_coord - start_centroid
        start_tangent -= np.dot(start_tangent, start_normal) * start_normal
        start_tangent /= np.linalg.norm(start_tangent)

        end_tangent = end_centroid - end_coord  # reversed direction for end tangent
        end_tangent -= np.dot(end_tangent, end_normal) * end_normal
        end_tangent /= np.linalg.norm(end_tangent)

        if start_end_centroid:
            # move start and end points towards their centroid
            # and then onto the plane formed by the centroid and
            # the ring normal
            start_rib = 0.5 * (start_centroid + start_coord)
            start_rib -= np.dot(start_rib - start_centroid, start_normal) * start_normal

            end_rib = 0.5 * (end_centroid + end_coord)
            end_rib -= np.dot(end_rib - end_centroid, end_normal) * end_normal
        else:
            start_rib = start_coord
            end_rib = end_coord

        rib_delta = start_rib - end_rib
        rib_interval = np.linalg.norm(rib_delta)
        rib_inc = rib_interval / rib_steps

        ftmp1 = 1.0 / rib_interval

        spline_a = np.zeros(3, dtype=np.float32)
        spline_a += 2 * ftmp1 * rib_delta
        spline_a += start_tangent
        spline_a += end_tangent
        spline_a *= ftmp1 * ftmp1

        spline_b = np.zeros(3, dtype=np.float32)
        spline_b -= 3 * ftmp1 * rib_delta
        spline_b -= 2 * start_tangent
        spline_b -= 1 * end_tangent
        spline_b *= ftmp1

        spline_c = start_tangent
        spline_d = start_rib

        # Create ribbon tangent spline
        tangent_a = np.zeros(3, dtype=np.float32)
        tangent_b = 3 * spline_a
        tangent_c = 2 * spline_b
        tangent_d = spline_c

        frames: list[RibbonFrame] = []

        # Initial frame
        f_right = np.cross(start_tangent, start_normal)
        f_right /= np.linalg.norm(f_right)
        frame = RibbonFrame(
            start_tangent,
            f_right,
            np.cross(f_right, start_tangent),
            start_rib,
            0.0,
        )
        frames.append(frame)

        t = rib_inc
        for _ in range(rib_steps):
            prev_frame = frames[-1]

            new_tangent = ribbon_spline(tangent_a, tangent_b, tangent_c, tangent_d, t)
            new_tangent /= np.linalg.norm(new_tangent)

            rot_axis = np.cross(prev_frame.forward, new_tangent)
            axis_norm = np.linalg.norm(rot_axis)

            # copy previous frame
            frame = prev_frame.copy()
            frame.origin = ribbon_spline(spline_a, spline_b, spline_c, spline_d, t)
            frame.arclength += np.linalg.norm(frame.origin - prev_frame.origin)

            # rotate frame if tangents not parallel
            if axis_norm > min_axis_norm:
                rot_angle = np.arccos(np.dot(prev_frame.forward, new_tangent))

                # Rotate frame angle rot_angle about rot_axis
                frame.forward = rotate(frame.forward, rot_axis, rot_angle)
                frame.right = rotate(frame.right, rot_axis, rot_angle)
                frame.up = rotate(frame.up, rot_axis, rot_angle)

            frames.append(frame)
            t += rib_inc

        start_right = frames[0].right
        end_right = np.cross(end_tangent, end_normal)
        end_right /= np.linalg.norm(end_right)

        correction_angle = np.arccos(np.dot(end_right, frames[-1].right))
        if np.dot(np.cross(end_right, frames[-1].right), end_tangent) > 0:
            correction_angle *= -1

        inc_angle = correction_angle / rib_steps
        curr_angle = -inc_angle

        top_color = np.array([128, 128, 255, 255], dtype=np.uint8)
        bottom_color = np.array([230, 230, 230, 255], dtype=np.uint8)

        triangle_offset = len(vertices)

        for i, frame in enumerate(frames):
            curr_angle += inc_angle

            # Apply correcting rotation:
            # Rotate frame angle curr_angle about frame.forward

            frame.right = rotate(frame.right, frame.forward, curr_angle)
            frame.up = rotate(frame.up, frame.forward, curr_angle)

            # vertices (of this frame's rectangle)

            # top right (index +0)
            vert = frame.origin + rib_height * frame.up + rib_width * frame.right
            vertices.append(vert)
            colors.append(top_color)
            norm = frame.up + frame.right
            norm /= np.linalg.norm(norm)
            normals.append(norm)

            # bottom right (index +1)
            vert = frame.origin - rib_height * frame.up + rib_width * frame.right
            vertices.append(vert)
            colors.append(bottom_color)
            norm = -frame.up + frame.right
            norm /= np.linalg.norm(norm)
            normals.append(norm)

            # bottom left (index +2)
            vert = frame.origin - rib_height * frame.up - rib_width * frame.right
            vertices.append(vert)
            colors.append(bottom_color)
            norm = -frame.up - frame.right
            norm /= np.linalg.norm(norm)
            normals.append(norm)

            # top left (index +3)
            vert = frame.origin + rib_height * frame.up - rib_width * frame.right
            vertices.append(vert)
            colors.append(top_color)
            norm = frame.up - frame.right
            norm /= np.linalg.norm(norm)
            normals.append(norm)

            if i == len(frames) - 1:
                continue  # no triangles for the last frame

            current_vertex_offset = triangle_offset + i * 4
            next_vertex_offset = triangle_offset + (i + 1) * 4

            # top 1
            triangles.append(
                (
                    current_vertex_offset + 0,  # current, top right
                    next_vertex_offset + 0,  # next, top right
                    current_vertex_offset + 3,  # current, top left
                )
            )

            # top 2
            triangles.append(
                (
                    next_vertex_offset + 0,  # next, top right
                    next_vertex_offset + 3,  # next, top left
                    current_vertex_offset + 3,  # current, top left
                )
            )

            # bottom 1
            triangles.append(
                (
                    current_vertex_offset + 1,  # current, bottom right
                    current_vertex_offset + 2,  # current, bottom left
                    next_vertex_offset + 1,  # next, bottom right
                )
            )

            # bottom 2
            triangles.append(
                (
                    next_vertex_offset + 1,  # next, bottom right
                    current_vertex_offset + 2,  # current, bottom left
                    next_vertex_offset + 2,  # next, bottom left
                )
            )

            # right 1
            triangles.append(
                (
                    current_vertex_offset + 0,  # current, top right
                    current_vertex_offset + 1,  # current, bottom right
                    next_vertex_offset + 0,  # next, top right
                )
            )

            # right 2
            triangles.append(
                (
                    next_vertex_offset + 0,  # next, top right
                    current_vertex_offset + 1,  # current, bottom right
                    next_vertex_offset + 1,  # next, bottom right
                )
            )

            # left 1
            triangles.append(
                (
                    current_vertex_offset + 3,  # current, top left
                    next_vertex_offset + 3,  # next, top left
                    current_vertex_offset + 2,  # current, bottom left
                )
            )

            # left 2
            triangles.append(
                (
                    next_vertex_offset + 3,  # next, top left
                    next_vertex_offset + 2,  # next, bottom left
                    current_vertex_offset + 2,  # current, bottom left
                )
            )

        if start_end_centroid:
            # Draw extensions of ribbon to meet hexagonal disks
            twister_draw_ribbon_extensions(
                vertices,
                colors,
                normals,
                triangles,
                start_centroid,
                start_normal,
                start_right,
                start_rib,
                rib_height,
                rib_width,
                top_color,
                bottom_color,
            )
            twister_draw_ribbon_extensions(
                vertices,
                colors,
                normals,
                triangles,
                end_centroid,
                end_normal,
                end_right,
                end_rib,
                rib_height,
                rib_width,
                top_color,
                bottom_color,
            )

            # Draw hexagonal disks for joining rings
            # XXX: do this only once per ring
            twister_draw_hexagon(
                vertices,
                colors,
                normals,
                triangles,
                start_centroid,
                start_normal,
                start_ring_coords[0],
                rib_height,
                rib_width,
                top_color,
                bottom_color,
            )

            twister_draw_hexagon(
                vertices,
                colors,
                normals,
                triangles,
                end_centroid,
                end_normal,
                end_ring_coords[0],
                rib_height,
                rib_width,
                top_color,
                bottom_color,
            )
        else:
            # Draw start and end end caps
            # Start end caps
            current_vertex_offset = triangle_offset

            triangles.append(
                (
                    current_vertex_offset + 3,  # top left
                    current_vertex_offset + 0,  # top right
                    current_vertex_offset + 1,  # bottom right
                )
            )

            triangles.append(
                (
                    current_vertex_offset + 2,  # bottom left
                    current_vertex_offset + 1,  # bottom right
                    current_vertex_offset + 3,  # top left
                )
            )

            # End end caps
            current_vertex_offset = triangle_offset + rib_steps * 4

            triangles.append(
                (
                    current_vertex_offset + 3,  # top left
                    current_vertex_offset + 0,  # top right
                    current_vertex_offset + 1,  # bottom right
                )
            )

            triangles.append(
                (
                    current_vertex_offset + 2,  # bottom left
                    current_vertex_offset + 1,  # bottom right
                    current_vertex_offset + 3,  # top left
                )
            )

    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    triangles = np.array(triangles, dtype=np.int32)

    # twister_drawing.display_style = Drawing.Mesh
    twister_drawing.set_geometry(vertices, normals, triangles)
    twister_drawing.set_vertex_colors(colors)


class TwisterDrawing(Drawing):
    def __init__(self, name="twister"):
        super().__init__(name)

    def clear(self):
        self.set_geometry(None, None, None)
        self.remove_all_drawings()

    @time
    def compute_twister(self, structure: Structure):
        _make_twister_graphics(structure, self)
