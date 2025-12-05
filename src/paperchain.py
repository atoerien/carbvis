from __future__ import annotations

from typing import Callable

import numpy as np
from chimerax.atomic import AtomicStructures, Atoms
from chimerax.core.colors import Color
from chimerax.core.models import MODEL_COLOR_CHANGED, Model, PickedModel
from chimerax.core.session import Session
from chimerax.graphics import Drawing, PickedTriangle, PickedTriangles, Texture

from ._paperchain import (  # pyright: ignore[reportMissingModuleSource]
    update_graphics,
)
from .carbs import CarbRing, PickedRing, PickedRings, find_rings
from .model import CarbVisModel
from .utils import IntArray, UByteArray


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
        cmap: Callable[[CarbRing], Color] | Color,
        tex_formula: str | None,
        tex_period: int,
        tex_duty: float,
    ):
        super().__init__(session, atoms, name, update=update)

        self.selection_coupled = AtomicStructures([self.structure])

        self.bipyramid_height = bipyramid_height
        self.max_ring_size = max_ring_size
        self.cmap = cmap

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
        cmap: Callable[[CarbRing], Color] | Color,
        tex_formula: str | None,
        tex_period: int,
        tex_duty: float,
    ):
        self.auto_update = update

        self.bipyramid_height = bipyramid_height
        self.max_ring_size = max_ring_size
        self.cmap = cmap

        self.tex_formula = tex_formula
        self.tex_period = tex_period
        self.tex_duty = tex_duty
        self._update_texture()

        self.rings = None
        self._clear_geometry()

    @line_profile
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

    def _get_overall_color(self):
        if isinstance(self.cmap, Color):
            return self.cmap
        return None

    def _set_overall_color(self, color):
        self.cmap = Color(color)
        self.session.triggers.activate_trigger(MODEL_COLOR_CHANGED, self)
        self._update_graphics()

    overall_color = model_color = property(_get_overall_color, _set_overall_color)

    @line_profile
    def _do_update(self, *, structure_changed, coords_changed):
        if structure_changed or self.rings is None:
            self._update_rings()
        if coords_changed or self.vertices is None:
            self._update_graphics()

    @line_profile
    def _update_rings(self):
        rings = find_rings(self.atoms, self.max_ring_size)

        if self.rings != rings:
            self.rings = rings
            self._clear_geometry()

    def _update_graphics(self):
        update_graphics(self)

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

    @line_profile
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
            "cmap": self.cmap,
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
            cmap=data["cmap"],
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
