from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from chimerax.atomic import Structure
from chimerax.atomic.changes import Changes
from chimerax.core.session import Session

from .carbs import CarbLinkage, find_linkages, find_rings
from .model import CarbVisModel
from .utils import FloatArray, Frame, color_float_to_ubyte, rotate, spline, time

# stop complaining when assigning float32 to float
if TYPE_CHECKING:
    float = float | np.floating


def draw_hexagon(
    vertices: list[FloatArray],
    normals: list[FloatArray],
    triangles: list[tuple[int, int, int]],
    vcolors: list[FloatArray],
    origin: FloatArray,
    right: FloatArray,
    up: FloatArray,
    rib_height: float,
    rib_width: float,
    top_color: FloatArray,
    bottom_color: FloatArray,
):
    triangle_offset = len(vertices)

    # top centroid vertex
    vert = origin + rib_height * up
    vertices.append(vert)
    vcolors.append(top_color)
    normals.append(up)

    # bottom centroid vertex
    vert = origin - rib_height * up
    vertices.append(vert)
    vcolors.append(bottom_color)
    normals.append(-up)

    # vertices for hexagon edges

    POLYGON_N = 12
    ROT_ANGLE = 2 * np.pi / POLYGON_N

    current_vec = right

    for i in range(POLYGON_N):
        # top hexagon vertex
        vert = origin + rib_height * up + rib_width * current_vec
        vertices.append(vert)
        vcolors.append(top_color)
        norm = up + current_vec
        norm /= np.linalg.norm(norm)
        normals.append(norm)

        # bottom hexagon vertex
        vert = origin - rib_height * up + rib_width * current_vec
        vertices.append(vert)
        vcolors.append(bottom_color)
        norm = -up + current_vec
        norm /= np.linalg.norm(norm)
        normals.append(norm)

        if i == POLYGON_N - 1:
            break  # don't bother rotating the last time

        # Rotate current vec pi/3 radians about the ring normal
        current_vec = rotate(current_vec, up, ROT_ANGLE)

    for i in range(POLYGON_N):
        edge1_top = triangle_offset + 2 + 2 * i
        edge1_bottom = edge1_top + 1
        if i < POLYGON_N - 1:
            edge2_top = edge1_top + 2
            edge2_bottom = edge1_top + 3
        else:
            edge2_top = triangle_offset + 2
            edge2_bottom = triangle_offset + 3

        # top
        triangles.append(
            (
                triangle_offset,
                edge1_top,
                edge2_top,
            )
        )

        # bottom
        triangles.append(
            (
                triangle_offset + 1,
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


class TwisterModel(CarbVisModel):
    def __init__(
        self,
        session: Session,
        structure: Structure,
        name: str | None = None,
        *,
        update: bool,
        start_end_centroid: bool,
        rib_steps: int,
        max_ring_size: int,
        rib_width: float,
        rib_height: float,
        colormap: Callable[[CarbLinkage], FloatArray] | None,
        gum_twist: bool,
    ):
        if name is None:
            name = f"{structure.name} Twister"
        super().__init__(session, structure, name, update=update)

        self.start_end_centroid = start_end_centroid
        self.rib_steps = rib_steps
        self.max_ring_size = max_ring_size
        self.rib_width = rib_width
        self.rib_height = rib_height
        self.dihedral_colormap = colormap
        self.gum_twist = gum_twist

    def update_params(
        self,
        *,
        update: bool,
        start_end_centroid: bool,
        rib_steps: int,
        max_ring_size: int,
        rib_width: float,
        rib_height: float,
        colormap: Callable[[CarbLinkage], FloatArray] | None,
        gum_twist: bool,
    ):
        if update != self.auto_update:
            self.auto_update = update

        clear_geometry = False
        if start_end_centroid != self.start_end_centroid:
            self.start_end_centroid = start_end_centroid
            clear_geometry = True
        if rib_steps != self.rib_steps:
            self.rib_steps = rib_steps
            clear_geometry = True
        if max_ring_size != self.max_ring_size:
            self.max_ring_size = max_ring_size
            clear_geometry = True
        if rib_width != self.rib_width:
            self.rib_width = rib_width
            clear_geometry = True
        if rib_height != self.rib_height:
            self.rib_height = rib_height
            clear_geometry = True
        if colormap != self.dihedral_colormap:
            self.dihedral_colormap = colormap
            clear_geometry = True
        if gum_twist != self.gum_twist:
            self.gum_twist = gum_twist
            clear_geometry = True

        if clear_geometry:
            self._clear_geometry()

    def _do_auto_update(self, changes: Changes):
        super()._do_auto_update(changes)

        # TODO: does this work for new bonds/atoms?
        if self._coordinates_changed(changes):
            self._recalculate_graphics()

    @time
    def _calc_graphics(self):
        start_end_centroid = self.start_end_centroid
        rib_steps = self.rib_steps
        rib_width = self.rib_width
        rib_height = self.rib_height
        colormap = self.dihedral_colormap
        gum_twist = self.gum_twist

        rings = find_rings(self.structure, self.max_ring_size)
        linkages = find_linkages(rings)

        # Threshold to determine when we have a reliable
        # rotation axis to rotate a frame about
        min_axis_norm = 1e-4

        vertices = []
        normals = []
        triangles = []
        vcolors = []

        disk_places = set()

        for link in linkages:
            start_ring = link.start_ring
            start_centroid, start_normal = start_ring.get_centroid_and_normal()

            end_ring = link.end_ring
            end_centroid, end_normal = end_ring.get_centroid_and_normal()

            start_coord = link.atoms[0].coord
            end_coord = link.atoms[-1].coord

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
                start_rib -= (
                    np.dot(start_rib - start_centroid, start_normal) * start_normal
                )

                end_rib = 0.5 * (end_centroid + end_coord)
                end_rib -= np.dot(end_rib - end_centroid, end_normal) * end_normal
            else:
                start_rib = start_coord
                end_rib = end_coord

            rib_delta = start_rib - end_rib
            rib_interval = np.linalg.norm(rib_delta)

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

            t = np.linspace(0, rib_interval, rib_steps + 1, dtype=np.float32)
            spath, stan = spline(spline_a, spline_b, spline_c, spline_d, t)

            if start_end_centroid:
                # extend the path to meet the centroids
                spath = np.vstack((start_centroid, spath, end_centroid))
                stan = np.vstack((start_tangent, stan, end_tangent))

            frames: list[Frame] = []

            # Initial frame
            point = spath[0]
            tangent = stan[0]
            up = start_normal
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

                    # Rotate frame angle rot_angle about rot_axis
                    frame.forward = rotate(frame.forward, rot_axis, rot_angle)
                    frame.right = rotate(frame.right, rot_axis, rot_angle)
                    frame.up = rotate(frame.up, rot_axis, rot_angle)

                frames.append(frame)

            if start_end_centroid:
                spline_frames = frames[1:-1]
            else:
                spline_frames = frames
            spline_end_frame = spline_frames[-1]

            end_right = np.cross(end_tangent, end_normal)
            end_right /= np.linalg.norm(end_right)

            correction_angle = np.arccos(np.dot(end_right, spline_end_frame.right))
            if np.dot(np.cross(end_right, spline_end_frame.right), end_tangent) > 0:
                correction_angle *= -1

            x = np.linspace(0, 1, rib_steps + 1, dtype=np.float32)
            if gum_twist:
                twist = correction_angle * np.sin(np.pi * x / 2) ** 2
            else:
                twist = correction_angle * x

            if start_end_centroid:
                # twist last frame the full angle to line up with the end ring
                frame = frames[-1]
                frame.right = rotate(frame.right, frame.forward, correction_angle)
                frame.up = rotate(frame.up, frame.forward, correction_angle)

            # rotate frames about frame.forward (twist)
            for i, frame in enumerate(spline_frames):
                rot_angle = twist[i]
                frame.right = rotate(frame.right, frame.forward, rot_angle)
                frame.up = rotate(frame.up, frame.forward, rot_angle)

            if gum_twist:
                x = np.linspace(0, 1, len(spline_frames), dtype=np.float32)
                wave = 1 - np.sin(np.pi * x) ** 4
                width_factor = 0.75 * abs(correction_angle) / np.pi
                width = rib_width * (width_factor * wave + (1 - width_factor))
            else:
                width = np.full(len(frames), rib_width, dtype=np.float32)

            if start_end_centroid:
                width = np.hstack((rib_width, width, rib_width))

            if colormap is not None:
                color = colormap(link)
                top_color = color
                bottom_color = color
            else:
                top_color = np.array([0.9, 0.9, 0.9], dtype=np.float32)
                bottom_color = np.array([0.5, 0.5, 1.0], dtype=np.float32)

            triangle_offset = len(vertices)

            for i, frame in enumerate(frames):
                w = width[i]

                # vertices (of this frame's rectangle)

                # top right (index +0)
                vert = frame.origin + rib_height * frame.up + w * frame.right
                vertices.append(vert)
                vcolors.append(top_color)
                norm = frame.up + frame.right
                norm /= np.linalg.norm(norm)
                normals.append(norm)

                # bottom right (index +1)
                vert = frame.origin - rib_height * frame.up + w * frame.right
                vertices.append(vert)
                vcolors.append(bottom_color)
                norm = -frame.up + frame.right
                norm /= np.linalg.norm(norm)
                normals.append(norm)

                # bottom left (index +2)
                vert = frame.origin - rib_height * frame.up - w * frame.right
                vertices.append(vert)
                vcolors.append(bottom_color)
                norm = -frame.up - frame.right
                norm /= np.linalg.norm(norm)
                normals.append(norm)

                # top left (index +3)
                vert = frame.origin + rib_height * frame.up - w * frame.right
                vertices.append(vert)
                vcolors.append(top_color)
                norm = frame.up - frame.right
                norm /= np.linalg.norm(norm)
                normals.append(norm)

                if i == len(frames) - 1:
                    continue  # no triangles for the last frame

                current_vertex_offset = triangle_offset + i * 4
                next_vertex_offset = current_vertex_offset + 4

                # top 1
                triangles.append(
                    (
                        current_vertex_offset,  # current, top right
                        next_vertex_offset,  # next, top right
                        current_vertex_offset + 3,  # current, top left
                    )
                )

                # top 2
                triangles.append(
                    (
                        next_vertex_offset,  # next, top right
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
                        current_vertex_offset,  # current, top right
                        current_vertex_offset + 1,  # current, bottom right
                        next_vertex_offset,  # next, top right
                    )
                )

                # right 2
                triangles.append(
                    (
                        next_vertex_offset,  # next, top right
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
                # Draw hexagonal disks for joining rings
                start_centroid_tuple = tuple(start_centroid)
                if start_centroid_tuple not in disk_places:
                    disk_places.add(start_centroid_tuple)

                    start_frame = frames[0]
                    draw_hexagon(
                        vertices,
                        normals,
                        triangles,
                        vcolors,
                        start_centroid,
                        start_frame.right,
                        start_frame.up,
                        rib_height,
                        rib_width,
                        top_color,
                        bottom_color,
                    )
                end_centroid_tuple = tuple(end_centroid)
                if end_centroid_tuple not in disk_places:
                    disk_places.add(end_centroid_tuple)

                    end_frame = frames[-1]
                    draw_hexagon(
                        vertices,
                        normals,
                        triangles,
                        vcolors,
                        end_centroid,
                        end_frame.right,
                        end_frame.up,
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
                        current_vertex_offset,  # top right
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
                        current_vertex_offset,  # top right
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

        va = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        na = np.array(normals, dtype=np.float32).reshape(-1, 3)
        ta = np.array(triangles, dtype=np.int32).reshape(-1, 3)
        ca = np.array(vcolors, dtype=np.float32).reshape(-1, 3)

        self.set_geometry(va, na, ta)
        self.set_vertex_colors(color_float_to_ubyte(ca))
