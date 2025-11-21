from __future__ import annotations

import numpy as np
from chimerax.atomic import AtomicStructures, Atoms
from chimerax.core.models import Model, PickedModel
from chimerax.core.session import Session
from chimerax.graphics import Drawing, PickedTriangle, PickedTriangles, Texture

from .carbs import CarbRing, PickedRing, PickedRings, find_rings, paperchain_colormap
from .model import CarbVisModel
from .utils import FloatArray, Frame, IntArray, UByteArray, color_float_to_ubyte, time


def make_texture(
    formula: str,
    size: int,
    period: int,
    duty: float,
    on_color: UByteArray,
    off_color: UByteArray,
) -> UByteArray:
    """
    Generate a procedural texture pattern.

    Args:
        formula: One of 'stripes', 'grid', 'diamond', 'rings', 'waves'.
        size: Texture size in pixels (square).
        period: Pixel distance between repeating features.
        duty: Fraction of each period that is 'off', [0, 1].
        on_color: (R, G, B) array for 'on' pixels, [0, 255].
        off_color: (R, G, B) tuple for 'off' pixels, [0, 255].
    """

    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size // 2, size // 2

    if formula == "stripes":
        mask = (y % period) < (period * duty)
    elif formula == "grid":
        mask_x = (x % period) < (0.5 * period * duty)
        mask_y = (y % period) < (0.5 * period * duty)
        mask = mask_x | mask_y
    elif formula == "diamond":
        d = np.abs(x - cx) + np.abs(y - cy)
        mask = (d % period) < (period * duty)
    elif formula == "rings":
        r = np.hypot(x - cx, y - cy)
        mask = (r % period) < (period * duty)
    elif formula == "waves":
        wave_y = y + np.sin(x / period * 2 * np.pi) * (0.5 * period * duty)
        mask = (wave_y % period) < (period * duty)
    else:
        raise ValueError(f"Unknown formula '{formula}'")

    tex = np.where(mask[..., None], off_color, on_color).astype(np.uint8)
    return tex


def ring_tex(ring_coords: FloatArray, frame: Frame) -> FloatArray:
    """Calculate ring texture coordinates."""

    n = len(ring_coords)

    # project coordinates down into the (forward, right) plane
    uv2d = np.empty((n, 2), dtype=np.float32)
    radii = np.empty(n, dtype=np.float32)
    for i, p in enumerate(ring_coords):
        dr = p - frame.origin
        dr_plane = dr - np.dot(dr, frame.up) * frame.up
        x = np.dot(dr_plane, frame.forward)
        y = np.dot(dr_plane, frame.right)
        uv2d[i] = (x, y)
        radii[i] = np.hypot(x, y)

    # scale and translate to [0, 1] texture space
    r_max = radii.max()
    if r_max == 0:
        # degenerate: all points at centre
        tex = np.full((n, 2), 0.5)
    else:
        tex = uv2d / (2.0 * r_max) + 0.5

    return tex


def draw_ring(
    vertices: list[FloatArray],
    normals: list[FloatArray],
    triangles: list[tuple[int, int, int]],
    vcolors: list[FloatArray],
    texcoords: list[tuple[float, float]],
    ring: CarbRing,
    color: FloatArray,
    bipyramid_height: float,
):
    """Draw the PaperChain polygon for a ring."""

    ring_coords = ring.coords
    frame = ring.get_frame()
    tex = ring_tex(ring_coords, frame)

    x = 0.5 * bipyramid_height * frame.up
    top = frame.origin + x
    bottom = frame.origin - x

    triangle_offset = len(vertices)

    vertices.append(top)
    normals.append(frame.up)
    vcolors.append(color)
    texcoords.append((0.5, 0.25))

    vertices.append(bottom)
    normals.append(-frame.up)
    vcolors.append(color)
    texcoords.append((0.5, 0.75))

    n = len(ring_coords)
    for i in range(n):
        next_i = (i + 1) % n

        curvec = ring_coords[i]
        nextvec = ring_coords[next_i]
        delta = nextvec - curvec

        normtop = np.cross(delta, curvec - top)
        normtop /= np.linalg.norm(normtop)

        # reverse normal direction for bottom half
        normbot = -np.cross(delta, curvec - bottom)
        normbot /= np.linalg.norm(normbot)

        texbase = tex[i]
        textop = (texbase[0], texbase[1] / 2)
        texbot = (texbase[0], texbase[1] / 2 + 0.5)

        vertices.append(curvec)
        normals.append(normtop)
        vcolors.append(color)
        texcoords.append(textop)

        triangles.append(
            (
                triangle_offset + 2 + (2 * next_i),
                triangle_offset + 2 + (2 * i),
                triangle_offset,
            )
        )

        vertices.append(curvec)
        normals.append(normbot)
        vcolors.append(color)
        texcoords.append(texbot)

        # order different to keep anticlockwise winding
        triangles.append(
            (
                triangle_offset + 3 + (2 * i),
                triangle_offset + 3 + (2 * next_i),
                triangle_offset + 1,
            )
        )


PAPERCHAIN_STATE_VERSION = 1


class PaperChainModel(CarbVisModel):
    def __init__(
        self,
        session: Session,
        atoms: Atoms,
        name: str,
        *,
        update: bool,
        bipyramid_height: float,
        max_ring_size: int,
        tex_formula: str | None,
        tex_period: int,
        tex_duty: float,
    ):
        super().__init__(session, atoms, name, update=update)

        self.selection_coupled = AtomicStructures([self.structure])

        self.bipyramid_height = bipyramid_height
        self.max_ring_size = max_ring_size

        self.rings: list[CarbRing] | None = None
        self.ring_to_triangle: IntArray | None = None
        self.triangle_to_ring: IntArray | None = None

        self.texture: Texture = Texture(linear_interpolation=True)
        self.tex_formula = tex_formula
        self.tex_period = tex_period
        self.tex_duty = tex_duty
        self._update_texture()

    def update_params(
        self,
        *,
        update: bool,
        bipyramid_height: float,
        max_ring_size: int,
        tex_formula: str | None,
        tex_period: int,
        tex_duty: float,
    ):
        self.auto_update = update

        self.bipyramid_height = bipyramid_height
        self.max_ring_size = max_ring_size

        self.tex_formula = tex_formula
        self.tex_period = tex_period
        self.tex_duty = tex_duty
        self._update_texture()

        self.rings = None
        self._clear_geometry()

    def _update_texture(self):
        """Regenerate and reload the texture."""

        on_color = np.full(3, 255, dtype=np.uint8)
        off_color = np.full(3, 128, dtype=np.uint8)

        if self.tex_formula is not None:
            tex = make_texture(
                formula=self.tex_formula,
                size=1024,
                period=self.tex_period,
                duty=self.tex_duty,
                on_color=on_color,
                off_color=off_color,
            )

            tex = np.concatenate(
                (
                    np.full((1024, 1024, 3), on_color, dtype=np.uint8),
                    tex,
                )
            )
        else:
            tex = np.full((32, 16, 3), on_color, dtype=np.uint8)

        self.texture.reload_texture(tex)

    def _clear_geometry(self):
        super()._clear_geometry()

        self.ring_to_triangle = None
        self.triangle_to_ring = None

        self.update_selection()

    def _do_update(self, *, structure_changed, coords_changed):
        if structure_changed or self.rings is None:
            self._update_rings()
        if coords_changed or self.vertices is None:
            self._update_graphics()

    @time
    def _update_rings(self):
        rings = find_rings(self.atoms, self.max_ring_size)

        if self.rings != rings:
            self.rings = rings
            self._clear_geometry()

    @time
    def _update_graphics(self):
        bipyramid_height = self.bipyramid_height

        vertices = []
        normals = []
        triangles = []
        vcolors = []
        texcoords = []
        r2t = []
        t2r = []

        assert self.rings is not None
        for i, ring in enumerate(self.rings):
            color = paperchain_colormap(ring)

            tri_before = len(triangles)
            draw_ring(
                vertices,
                normals,
                triangles,
                vcolors,
                texcoords,
                ring,
                color,
                bipyramid_height,
            )
            tri_after = len(triangles)

            r2t.append((tri_before, tri_after))
            t2r.extend(i for _ in range(tri_before, tri_after))

        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        triangles = np.array(triangles, dtype=np.int32)
        vcolors = np.array(vcolors, dtype=np.float32)
        texcoords = np.array(texcoords, dtype=np.float32)
        r2t = np.array(r2t, dtype=np.int32)
        t2r = np.array(t2r, dtype=np.int32)

        self.set_geometry(vertices, normals, triangles)
        self.ring_to_triangle = r2t
        self.triangle_to_ring = t2r
        self.set_vertex_colors(color_float_to_ubyte(vcolors))
        self.texture_coordinates = texcoords
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

        if rings is None or t2r is None:
            pa = PickedModel(self, p.distance)
        else:
            t = p.triangle_number
            ring = rings[t2r[t]]
            pa = PickedRing(ring, p.distance)

        setattr(pa, "triangle_pick", p)

        return pa

    def planes_pick(self, planes, exclude=None):
        rp = Drawing.planes_pick(self, planes, exclude)
        if self.triangles is None or not rp:
            return rp

        rings = self.rings
        r2t = self.ring_to_triangle

        if rings is None or r2t is None:
            return rp

        picks = []
        for p in rp:
            if isinstance(p, PickedTriangles) and p.drawing() is self:
                tmask = p._triangles_mask
                rr = [r for r, (ts, te) in zip(rings, r2t) if np.all(tmask[ts:te])]
                picks.append(PickedRings(rr))
        return picks

    def get_selected(self, include_children=False, fully=False):
        rings = self.rings

        if rings is None:
            return False

        if fully:
            for ring in rings:
                if not ring.selected:
                    return False
            if include_children:
                for c in self.child_models():
                    if not c.get_selected(include_children=True, fully=True):
                        return False
            return True

        for ring in rings:
            if ring.selected:
                return True
        if include_children:
            for c in self.child_models():
                if c.get_selected(include_children=True):
                    return True
        return False

    def set_selected(self, sel, *, fire_trigger=True):
        rings = self.rings

        if rings is not None:
            for ring in rings:
                ring.selected = sel
            self.update_selection(fire_trigger=fire_trigger)

    selected = property(get_selected, set_selected)

    @time
    def update_selection(self, *, fire_trigger=True):
        rings = self.rings
        t2r = self.triangle_to_ring

        if rings is None or t2r is None:
            self.highlighted_triangles_mask = None
            return

        ring_mask = np.empty(len(rings), dtype=bool)
        for i, ring in enumerate(rings):
            ring_mask[i] = ring.selected

        triangle_mask = ring_mask[t2r]
        self.highlighted_triangles_mask = triangle_mask
        if fire_trigger:
            self._selection_changed()

    # these attrs will all be recalculated on restore if auto-updating,
    # so we don't need to save them if auto_update is not set
    _save_noauto_attrs = (
        "vertices",
        "normals",
        "triangles",
        "rings",
        "ring_to_triangle",
        "triangle_to_ring",
        "vertex_colors",
        "texture_coordinates",
    )

    def take_snapshot(self, session: Session, flags: int):
        data = {
            "atoms": self.atoms,
            "name": self.name,
            "update": self.auto_update,
            "bipyramid_height": self.bipyramid_height,
            "max_ring_size": self.max_ring_size,
            "tex_formula": self.tex_formula,
            "tex_period": self.tex_period,
            "tex_duty": self.tex_duty,
        }
        data["model state"] = Model.take_snapshot(self, session, flags)
        if not self.auto_update:
            for attr in self._save_noauto_attrs:
                data[attr] = getattr(self, attr)
        data["version"] = PAPERCHAIN_STATE_VERSION
        return data

    @classmethod
    def restore_snapshot(cls, session: Session, data: dict):
        ret = PaperChainModel(
            session,
            data["atoms"],
            data["name"],
            update=data["update"],
            bipyramid_height=data["bipyramid_height"],
            max_ring_size=data["max_ring_size"],
            tex_formula=data["tex_formula"],
            tex_period=data["tex_period"],
            tex_duty=data["tex_duty"],
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
