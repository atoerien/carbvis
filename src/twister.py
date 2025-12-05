from __future__ import annotations

from typing import Callable

import numpy as np
from chimerax.atomic import AtomicStructures, Atoms
from chimerax.core.colors import Color
from chimerax.core.models import MODEL_COLOR_CHANGED, Model, PickedModel
from chimerax.core.session import Session
from chimerax.graphics import Drawing, PickedTriangle, PickedTriangles

from ._twister import (  # pyright: ignore[reportMissingModuleSource]
    update_graphics,
)
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
from .utils import IntArray

TWISTER_STATE_VERSION = 1


class TwisterModel(CarbVisModel):
    def __init__(
        self,
        session: Session,
        atoms: Atoms,
        name: str,
        *,
        update: bool,
        start_end_centroid: bool,
        rib_steps: int,
        max_ring_size: int,
        max_path_len: int,
        rib_width: float,
        rib_height: float,
        cmap_top: Callable[[CarbLinkage], Color] | Color,
        cmap_bottom: Callable[[CarbLinkage], Color] | Color,
        gum_twist: bool,
    ):
        super().__init__(session, atoms, name, update=update)

        self.selection_coupled = AtomicStructures([self.structure])

        self.start_end_centroid = start_end_centroid
        self.rib_steps = rib_steps
        self.max_ring_size = max_ring_size
        self.max_path_len = max_path_len
        self.rib_width = rib_width
        self.rib_height = rib_height
        self.cmap_top = cmap_top
        self.cmap_bottom = cmap_bottom
        self.gum_twist = gum_twist

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
        start_end_centroid: bool,
        rib_steps: int,
        max_ring_size: int,
        max_path_len: int,
        rib_width: float,
        rib_height: float,
        cmap_top: Callable[[CarbLinkage], Color] | Color,
        cmap_bottom: Callable[[CarbLinkage], Color] | Color,
        gum_twist: bool,
    ):
        self.auto_update = update

        self.start_end_centroid = start_end_centroid
        self.rib_steps = rib_steps
        self.max_ring_size = max_ring_size
        self.max_path_len = max_path_len
        self.rib_width = rib_width
        self.rib_height = rib_height
        self.cmap_top = cmap_top
        self.cmap_bottom = cmap_bottom
        self.gum_twist = gum_twist

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

    def _get_overall_color(self):
        if isinstance(self.cmap_top, Color):
            if self.cmap_top == self.cmap_bottom:
                return self.cmap_top
        return None

    def _set_overall_color(self, color):
        c = Color(color)
        self.cmap_top = c
        self.cmap_bottom = c
        self.session.triggers.activate_trigger(MODEL_COLOR_CHANGED, self)
        self._update_graphics()

    overall_color = model_color = property(_get_overall_color, _set_overall_color)

    @line_profile
    def _do_update(self, *, structure_changed, coords_changed):
        if structure_changed or self.linkages is None:
            self._update_linkages()
        if coords_changed or self.vertices is None:
            self._update_graphics()

    @line_profile
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

    @line_profile
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
            "start_end_centroid": self.start_end_centroid,
            "rib_steps": self.rib_steps,
            "max_ring_size": self.max_ring_size,
            "max_path_len": self.max_path_len,
            "rib_width": self.rib_width,
            "rib_height": self.rib_height,
            "cmap_top": self.cmap_top,
            "cmap_bottom": self.cmap_bottom,
            "gum_twist": self.gum_twist,
        }
        data["model state"] = Model.take_snapshot(self, session, flags)
        if not self.auto_update:
            for attr in self._save_noauto_attrs:
                data[attr] = getattr(self, attr)
        data["version"] = TWISTER_STATE_VERSION
        return data

    @classmethod
    def restore_snapshot(cls, session: Session, data: dict):
        ret = TwisterModel(
            session,
            data["atoms"],
            data["name"],
            update=data["update"],
            start_end_centroid=data["start_end_centroid"],
            rib_steps=data["rib_steps"],
            max_ring_size=data["max_ring_size"],
            max_path_len=data["max_path_len"],
            rib_width=data["rib_width"],
            rib_height=data["rib_height"],
            cmap_top=data["cmap_top"],
            cmap_bottom=data["cmap_bottom"],
            gum_twist=data["gum_twist"],
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
