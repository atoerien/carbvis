from __future__ import annotations

from typing import Callable

import numpy as np
from chimerax.atomic import AtomicStructures, Atoms
from chimerax.core.models import Model, PickedModel
from chimerax.core.session import Session
from chimerax.graphics import Drawing, PickedTriangle, PickedTriangles

from .carbs import (
    CarbLinkage,
    CarbRing,
    PickedLinkage,
    PickedLinkages,
    PickedRing,
    PickedRings,
    find_linkages,
    find_rings,
)
from .model import CarbVisModel
from .utils import (
    FloatArray,
    Frame,
    IntArray,
    color_float_to_ubyte,
    rotate,
    spherical_to_xyz,
    spline,
    time,
)


def draw_tube(
    vertices: list[FloatArray],
    normals: list[FloatArray],
    triangles: list[tuple[int, int, int]],
    vcolors: list[FloatArray],
    ring_spheres: dict[int, tuple[list[int], list[FloatArray]]],
    linkage: CarbLinkage,
    color_a: FloatArray,
    color_b: FloatArray,
    radius: float,
    candy_cane: bool,
    segment_subdivisions: int,
    circle_subdivisions: int,
):
    """Draw a tube along a linkage."""

    start_ring = linkage.start_ring
    start_centroid, start_normal = start_ring.get_centroid_and_normal()

    end_ring = linkage.end_ring
    end_centroid, end_normal = end_ring.get_centroid_and_normal()

    start_coord = linkage[0].coord
    end_coord = linkage[-1].coord

    start_tangent = start_coord - start_centroid
    start_tangent -= np.dot(start_tangent, start_normal) * start_normal
    start_tangent /= np.linalg.norm(start_tangent)

    end_tangent = end_centroid - end_coord  # reversed direction for end tangent
    end_tangent -= np.dot(end_tangent, end_normal) * end_normal
    end_tangent /= np.linalg.norm(end_tangent)

    # start the spline away from the ring centroids
    # will extend the tubes to meet the centroids later
    # this ensures the tube is not twisting
    # when it intersects with the spheres
    start_spline = start_centroid + radius * start_tangent
    end_spline = end_centroid - radius * end_tangent

    delta = start_spline - end_spline
    interval = np.linalg.norm(delta)

    ftmp1 = 1.0 / interval

    spline_a = np.zeros(3, dtype=np.float32)
    spline_a += 2 * ftmp1 * delta
    spline_a += start_tangent
    spline_a += end_tangent
    spline_a *= ftmp1 * ftmp1

    spline_b = np.zeros(3, dtype=np.float32)
    spline_b -= 3 * ftmp1 * delta
    spline_b -= 2 * start_tangent
    spline_b -= 1 * end_tangent
    spline_b *= ftmp1

    spline_c = start_tangent
    spline_d = start_spline

    t = np.linspace(0, interval, segment_subdivisions + 1, dtype=np.float32)
    spath, stan = spline(spline_a, spline_b, spline_c, spline_d, t)

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

    frames.append(Frame(point, up, tangent, right))

    for i in range(1, len(spath)):
        prev_frame = frames[-1]
        point = spath[i]
        tangent = stan[i]

        # copy previous frame
        frame = prev_frame.copy()
        frame.origin = point

        # align frame with tangent
        frame.align(tangent)

        frames.append(frame)

    if candy_cane:
        spline_end_frame = frames[-2]

        end_right = np.cross(end_tangent, end_normal)
        end_right /= np.linalg.norm(end_right)

        correction_angle = np.arccos(np.dot(end_right, spline_end_frame.right))
        if np.dot(np.cross(end_right, spline_end_frame.right), end_tangent) > 0:
            correction_angle *= -1

        x = np.linspace(0, 1, segment_subdivisions + 1, dtype=np.float32)
        twist = correction_angle * np.sin(np.pi * x / 2) ** 2

        # rotate frames about frame.forward (twist)
        # don't twist the first and last frames
        for i, frame in enumerate(frames[1:-1]):
            rot_angle = twist[i]
            frame.up = rotate(frame.up, frame.forward, rot_angle)
            frame.right = rotate(frame.right, frame.forward, rot_angle)

        # twist last frame the full angle to line up with the end ring
        frame = frames[-1]
        frame.up = rotate(frame.up, frame.forward, correction_angle)
        frame.right = rotate(frame.right, frame.forward, correction_angle)

    triangle_offset = len(vertices)

    for i, frame in enumerate(frames):
        a, step = np.linspace(
            0,
            2 * np.pi,
            circle_subdivisions,
            endpoint=False,
            retstep=True,
            dtype=np.float32,
        )
        # align the middle of the first stripe with frame.up
        a -= step / 2

        n = np.cos(a).reshape(-1, 1) * frame.up - np.sin(a).reshape(-1, 1) * frame.right
        v = frame.origin + radius * n

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
            offset = triangle_offset + (2 * i) * circle_subdivisions
            offset_next = offset + 2 * circle_subdivisions
            for j in range(0, circle_subdivisions, 2):
                k0 = offset + j
                k1 = offset + (j + 1) % circle_subdivisions
                k2 = offset_next + j
                k3 = offset_next + (j + 1) % circle_subdivisions
                triangles.append((k0, k2, k1))
                triangles.append((k1, k2, k3))

            offset = triangle_offset + (2 * i + 1) * circle_subdivisions
            offset_next = offset + 2 * circle_subdivisions
            for j in range(1, circle_subdivisions, 2):
                k0 = offset + j
                k1 = offset + (j + 1) % circle_subdivisions
                k2 = offset_next + j
                k3 = offset_next + (j + 1) % circle_subdivisions
                triangles.append((k0, k2, k1))
                triangles.append((k1, k2, k3))
        else:
            offset = triangle_offset + i * circle_subdivisions
            offset_next = offset + circle_subdivisions
            for j in range(circle_subdivisions):
                k0 = offset + j
                k1 = offset + (j + 1) % circle_subdivisions
                k2 = offset_next + j
                k3 = offset_next + (j + 1) % circle_subdivisions
                triangles.append((k0, k2, k1))
                triangles.append((k1, k2, k3))

    for i, ring in zip(
        (0, len(frames) - 1),
        (start_ring, end_ring),
    ):
        if candy_cane:
            offset = triangle_offset + (2 * i) * circle_subdivisions
        else:
            offset = triangle_offset + i * circle_subdivisions

        id_ring = id(ring)
        if id_ring not in ring_spheres:
            ring_spheres[id_ring] = ([offset], [color_a])
        else:
            offset_list, color_list = ring_spheres[id_ring]
            offset_list.append(offset)
            color_list.append(color_a)


def draw_sphere(
    vertices: list[FloatArray],
    normals: list[FloatArray],
    triangles: list[tuple[int, int, int]],
    vcolors: list[FloatArray],
    ring: CarbRing,
    color_a: FloatArray,
    color_b: FloatArray,
    radius: float,
    candy_cane: bool,
    circle_subdivisions,
):
    """Draw a sphere at an ring."""

    origin, up = ring.get_centroid_and_normal()

    n_lon = 128
    lat_inc = np.pi / circle_subdivisions
    n_lat = circle_subdivisions // 2 + 2

    # exclude theta and pi for theta, poles handled separately
    phi = np.linspace(0, 2 * np.pi, n_lon, endpoint=False, dtype=np.float32)
    theta = np.linspace(lat_inc, np.pi - lat_inc, n_lat - 2, dtype=np.float32)

    phi, theta = np.meshgrid(phi, theta)
    phi = phi.reshape(-1)
    theta = theta.reshape(-1)
    n = np.stack((np.full(len(theta), 1), theta, phi), axis=-1)
    n = spherical_to_xyz(n)

    # use the z-component of the norm of the first latitude line
    # to scale the radius for the poles, so that the top and bottom
    # "faces" of the sphere are flat, to align with the tubes
    # HACK: multiply by 0.99 to try avoid z-fighting
    pole_radius = 0.99 * radius * n[0][2]

    n_top = np.array([0, 0, 1])
    n_bot = np.array([0, 0, -1])

    # align top vector of sphere to up

    rot_axis = np.cross(n_top, up)
    axis_norm = np.linalg.norm(rot_axis)

    # skip rotation if we're already aligned
    if axis_norm > 1e-4:
        rot_angle = np.arccos(
            # float shenanigans
            np.clip(np.dot(n_top, up), -1.0, 1.0)
        )

        n_top = rotate(n_top, rot_axis, rot_angle)
        n = rotate(n, rot_axis, rot_angle)
        n_bot = rotate(n_bot, rot_axis, rot_angle)

    v_top = origin + pole_radius * n_top
    v = origin + radius * n
    v_bot = origin + pole_radius * n_bot

    triangle_offset = len(vertices)

    vertices.append(v_top)
    normals.append(n_top)
    vcolors.append(color_a)

    vertices.extend(v)
    normals.extend(n)
    vcolors.extend(color_a for _ in range(len(v)))

    if candy_cane:
        vertices.extend(v)
        normals.extend(n)
        vcolors.extend(color_b for _ in range(len(v)))

    vertices.append(v_bot)
    normals.append(n_bot)
    vcolors.append(color_a)

    k0 = triangle_offset
    for i in range(n_lon):
        k1 = triangle_offset + 1 + i
        k2 = triangle_offset + 1 + (i + 1) % n_lon
        triangles.append((k0, k1, k2))

    if candy_cane:
        for i in range(0, n_lat - 3, 2):
            row_start = triangle_offset + 1 + ((n_lat - 2) + i) * n_lon
            next_row_start = row_start + n_lon
            for j in range(n_lon):
                k0 = row_start + j
                k1 = row_start + (j + 1) % n_lon
                k2 = next_row_start + j
                k3 = next_row_start + (j + 1) % n_lon
                triangles.append((k0, k2, k1))
                triangles.append((k1, k2, k3))

        for i in range(1, n_lat - 3, 2):
            row_start = triangle_offset + 1 + i * n_lon
            next_row_start = row_start + n_lon
            for j in range(n_lon):
                k0 = row_start + j
                k1 = row_start + (j + 1) % n_lon
                k2 = next_row_start + j
                k3 = next_row_start + (j + 1) % n_lon
                triangles.append((k0, k2, k1))
                triangles.append((k1, k2, k3))
    else:
        for i in range(n_lat - 3):
            row_start = triangle_offset + 1 + i * n_lon
            next_row_start = row_start + n_lon
            for j in range(n_lon):
                k0 = row_start + j
                k1 = row_start + (j + 1) % n_lon
                k2 = next_row_start + j
                k3 = next_row_start + (j + 1) % n_lon
                triangles.append((k0, k2, k1))
                triangles.append((k1, k2, k3))

    base = triangle_offset + 1 + (n_lat - 3) * n_lon
    k0 = len(vertices) - 1
    for i in range(n_lon):
        k1 = base + i
        k2 = base + (i + 1) % n_lon
        triangles.append((k0, k2, k1))


STRAND_STATE_VERSION = 1


class StrandModel(CarbVisModel):
    def __init__(
        self,
        session: Session,
        atoms: Atoms,
        name: str,
        *,
        update: bool,
        max_ring_size: int,
        max_path_len: int,
        radius: float,
        colormap: Callable[[CarbLinkage], FloatArray],
        candy_cane: bool,
        sphere_radius: float,
        sphere_colormap: Callable[[CarbRing], FloatArray] | None,
    ):
        super().__init__(session, atoms, name, update=update)

        self.selection_coupled = AtomicStructures([self.structure])

        self.max_ring_size = max_ring_size
        self.max_path_len = max_path_len
        self.radius = radius
        self.dihedral_colormap = colormap
        self.candy_cane = candy_cane
        self.sphere_radius = sphere_radius
        self.sphere_colormap = sphere_colormap

        # only rings included in one or more linkages
        self.rings: list[CarbRing] | None = None
        self.ring_to_triangle: IntArray | None = None
        self.triangle_to_ring: IntArray | None = None

        self.linkages: list[CarbLinkage] | None = None
        self.link_to_triangle: IntArray | None = None
        self.triangle_to_link: IntArray | None = None

    def update_params(
        self,
        *,
        update: bool,
        max_ring_size: int,
        max_path_len: int,
        radius: float,
        colormap: Callable[[CarbLinkage], FloatArray],
        candy_cane: bool,
        sphere_radius: float,
        sphere_colormap: Callable[[CarbRing], FloatArray] | None,
    ):
        self.auto_update = update

        self.max_ring_size = max_ring_size
        self.max_path_len = max_path_len
        self.radius = radius
        self.dihedral_colormap = colormap
        self.candy_cane = candy_cane
        self.sphere_radius = sphere_radius
        self.sphere_colormap = sphere_colormap

        self.rings = None
        self.linkages = None

        self._clear_geometry()

    def _clear_geometry(self):
        super()._clear_geometry()

        self.ring_to_triangle = None
        self.triangle_to_ring = None
        self.link_to_triangle = None
        self.triangle_to_link = None

        self.update_selection()

    def _do_update(self, *, structure_changed, coords_changed):
        if structure_changed or self.linkages is None:
            self._update_linkages()
        if coords_changed or self.vertices is None:
            self._update_graphics()

    @time
    def _update_linkages(self):
        rings = find_rings(self.atoms, self.max_ring_size)
        linkages = find_linkages(rings, self.max_path_len)

        if self.linkages != linkages:
            rings_set = {
                id(r) for link in linkages for r in (link.start_ring, link.end_ring)
            }
            self.rings = [r for r in rings if id(r) in rings_set]

            self.linkages = linkages
            self._clear_geometry()

    @time
    def _update_graphics(self):
        radius = self.radius
        colormap = self.dihedral_colormap
        candy_cane = self.candy_cane
        sphere_radius = self.sphere_radius
        sphere_colormap = self.sphere_colormap

        vertices = []
        normals = []
        triangles = []
        vcolors = []
        r2t = []
        t2r = []
        l2t = []
        t2l = []

        segment_subdivisions = 20
        circle_subdivisions = 16

        color_b = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        ring_spheres: dict[int, tuple[list[int], list[FloatArray]]] = {}

        assert self.linkages is not None
        for i, link in enumerate(self.linkages):
            color_a = colormap(link)

            tri_before = len(triangles)
            draw_tube(
                vertices,
                normals,
                triangles,
                vcolors,
                ring_spheres,
                link,
                color_a,
                color_b,
                radius,
                candy_cane,
                segment_subdivisions,
                circle_subdivisions,
            )
            tri_after = len(triangles)

            l2t.append((tri_before, tri_after))
            t2l.extend(i for _ in range(tri_before, tri_after))
            t2r.extend(-1 for _ in range(tri_before, tri_after))

        sphere_radius = max(sphere_radius, radius)

        if sphere_radius == radius:
            sphere_candy_cane = candy_cane
            sphere_colormap = None
        else:
            sphere_candy_cane = False

        assert self.rings is not None
        for i, ring in enumerate(self.rings):
            offset_list, color_list = ring_spheres[id(ring)]

            if sphere_colormap is not None:
                color_a = sphere_colormap(ring)
            else:
                color_a = np.average(color_list, axis=0)

                if sphere_radius == radius:
                    # overwrite vcolors for each adjacent frame
                    # this blends colors through the ring
                    for offset in offset_list:
                        for i in range(circle_subdivisions):
                            vcolors[offset + i] = color_a

            tri_before = len(triangles)
            draw_sphere(
                vertices,
                normals,
                triangles,
                vcolors,
                ring,
                color_a,
                color_b,
                sphere_radius,
                sphere_candy_cane,
                circle_subdivisions,
            )
            tri_after = len(triangles)

            r2t.append((tri_before, tri_after))
            t2r.extend(i for _ in range(tri_before, tri_after))
            t2l.extend(-1 for _ in range(tri_before, tri_after))

        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        triangles = np.array(triangles, dtype=np.int32)
        vcolors = np.array(vcolors, dtype=np.float32)
        r2t = np.array(r2t, dtype=np.int32)
        t2r = np.array(t2r, dtype=np.int32)
        l2t = np.array(l2t, dtype=np.int32)
        t2l = np.array(t2l, dtype=np.int32)

        self.set_geometry(vertices, normals, triangles)
        self.ring_to_triangle = r2t
        self.triangle_to_ring = t2r
        self.link_to_triangle = l2t
        self.triangle_to_link = t2l
        self.set_vertex_colors(color_float_to_ubyte(vcolors))
        self.update_selection()

    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        p = Drawing.first_intercept(self, mxyz1, mxyz2, exclude)
        if (
            self.triangles is None
            or not isinstance(p, PickedTriangle)
            or p.drawing() is not self
        ):
            return p

        rings = self.rings
        t2r = self.triangle_to_ring
        linkages = self.linkages
        t2l = self.triangle_to_link

        if rings is None or t2r is None or linkages is None or t2l is None:
            pa = PickedModel(self, p.distance)
        else:
            t = p.triangle_number
            r = t2r[t]
            l = t2l[t]
            if r != -1:
                ring = rings[r]
                pa = PickedRing(ring, p.distance)
            elif l != -1:
                linkage = linkages[l]
                pa = PickedLinkage(linkage, p.distance)
            else:
                pa = PickedModel(self, p.distance)

        setattr(pa, "triangle_pick", p)

        return pa

    def planes_pick(self, planes, exclude=None):
        rp = Drawing.planes_pick(self, planes, exclude)
        if self.triangles is None or not rp:
            return rp

        rings = self.rings
        r2t = self.ring_to_triangle
        linkages = self.linkages
        l2t = self.link_to_triangle

        if rings is None or r2t is None or linkages is None or l2t is None:
            return rp

        picks = []
        for p in rp:
            if isinstance(p, PickedTriangles) and p.drawing() is self:
                tmask = p._triangles_mask
                rr = [r for r, (ts, te) in zip(rings, r2t) if np.all(tmask[ts:te])]
                picks.append(PickedRings(rr))
                ll = [l for l, (ts, te) in zip(linkages, l2t) if np.all(tmask[ts:te])]
                picks.append(PickedLinkages(ll))
        return picks

    def get_selected(self, include_children=False, fully=False):
        rings = self.rings
        linkages = self.linkages

        if rings is None or linkages is None:
            return False

        if fully:
            for ring in rings:
                if not ring.selected:
                    return False
            for link in linkages:
                if not link.selected:
                    return False
            if include_children:
                for c in self.child_models():
                    if not c.get_selected(include_children=True, fully=True):
                        return False
            return True

        for ring in rings:
            if ring.selected:
                return True
        for link in linkages:
            if link.selected:
                return True
        if include_children:
            for c in self.child_models():
                if c.get_selected(include_children=True):
                    return True
        return False

    def set_selected(self, sel, *, fire_trigger=True):
        rings = self.rings
        linkages = self.linkages

        if rings is not None and linkages is not None:
            for ring in rings:
                ring.selected = sel
            for link in linkages:
                link.selected = sel
            self.update_selection(fire_trigger=fire_trigger)

    selected = property(get_selected, set_selected)

    def update_selection(self, *, fire_trigger=True):
        rings = self.rings
        t2r = self.triangle_to_ring
        linkages = self.linkages
        t2l = self.triangle_to_link

        if rings is None or t2r is None or linkages is None or t2l is None:
            self.highlighted_triangles_mask = None
            return

        ring_mask = np.empty(len(rings) + 1, dtype=bool)
        for i, ring in enumerate(rings):
            ring_mask[i] = ring.selected
        ring_mask[-1] = False  # use False for -1 in t2r

        link_mask = np.empty(len(linkages) + 1, dtype=bool)
        for i, link in enumerate(linkages):
            link_mask[i] = link.selected
        link_mask[-1] = False  # use False for -1 in t2l

        triangle_mask = ring_mask[t2r]
        triangle_mask |= link_mask[t2l]

        self.highlighted_triangles_mask = triangle_mask
        if fire_trigger:
            self._selection_changed()

    # these attrs will all be recalculated on restore if auto-updating,
    # so we don't need to save them if auto_update is not set
    _save_noauto_attrs = (
        "vertices",
        "normals",
        "triangles",
        "ring",
        "ring_to_triangle",
        "triangle_to_ring",
        "linkages",
        "link_to_triangle",
        "triangle_to_link",
        "vertex_colors",
    )

    def take_snapshot(self, session: Session, flags: int):
        data = {
            "atoms": self.atoms,
            "name": self.name,
            "update": self.auto_update,
            "max_ring_size": self.max_ring_size,
            "max_path_len": self.max_path_len,
            "radius": self.radius,
            "colormap": self.colormap,
            "candy_cane": self.candy_cane,
            "sphere_radius": self.sphere_radius,
            "sphere_colormap": self.sphere_colormap,
        }
        data["model state"] = Model.take_snapshot(self, session, flags)
        if not self.auto_update:
            for attr in self._save_noauto_attrs:
                data[attr] = getattr(self, attr)
        data["version"] = STRAND_STATE_VERSION
        return data

    @classmethod
    def restore_snapshot(cls, session: Session, data: dict):
        ret = StrandModel(
            session,
            data["atoms"],
            data["name"],
            update=data["update"],
            max_ring_size=data["max_ring_size"],
            max_path_len=data["max_path_len"],
            radius=data["radius"],
            colormap=data["colormap"],
            candy_cane=data["candy_cane"],
            sphere_radius=data["sphere_radius"],
            sphere_colormap=data["sphere_colormap"],
        )
        ret.set_state_from_snapshot(session, data)
        return ret

    def set_state_from_snapshot(self, session, data):
        Model.set_state_from_snapshot(self, session, data["model state"])
        if not self.auto_update:
            # need to do vertices, normals, triangles first
            self.set_geometry(data["vertices"], data["normals"], data["triangles"])
            for attr in self._save_noauto_attrs[3:]:
                setattr(self, attr, data[attr])
