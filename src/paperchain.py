from __future__ import annotations

import numpy as np
from chimerax.atomic import Atoms
from chimerax.atomic.changes import Changes
from chimerax.core.models import Model
from chimerax.core.session import Session
from chimerax.graphics import Texture

from .carbs import find_rings, paperchain_colormap
from .model import CarbVisModel
from .utils import FloatArray, Frame, UByteArray, color_float_to_ubyte, time


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

        self.bipyramid_height = bipyramid_height
        self.max_ring_size = max_ring_size

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
        if update != self.auto_update:
            self.auto_update = update

        clear_geometry = False
        if bipyramid_height != self.bipyramid_height:
            self.bipyramid_height = bipyramid_height
            clear_geometry = True
        if max_ring_size != self.max_ring_size:
            self.max_ring_size = max_ring_size
            clear_geometry = True

        update_texture = False
        if tex_formula != self.tex_formula:
            self.tex_formula = tex_formula
            update_texture = True
        if tex_period != self.tex_period:
            self.tex_period = tex_period
            update_texture = True
        if tex_duty != self.tex_duty:
            self.tex_duty = tex_duty
            update_texture = True

        if clear_geometry:
            self._clear_geometry()
        if update_texture:
            self._update_texture()

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

    def _do_auto_update(self, changes: Changes):
        super()._do_auto_update(changes)

        if self._structure_changed(changes):
            self._recalculate_graphics()

    @time
    def _calc_graphics(self):
        bipyramid_height = self.bipyramid_height

        vertices = []
        normals = []
        triangles = []
        vcolors = []
        texcoords = []

        rings = find_rings(self.atoms, self.max_ring_size)

        for ring in rings:
            n = len(ring.atoms)

            color = paperchain_colormap(ring)

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

            for i in range(n):
                # calculate next ring position (wrapping as necessary)
                next_i = i + 1
                if next_i >= n:
                    next_i = 0

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

        va = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        na = np.array(normals, dtype=np.float32).reshape(-1, 3)
        ta = np.array(triangles, dtype=np.int32).reshape(-1, 3)
        ca = np.array(vcolors, dtype=np.float32).reshape(-1, 3)
        tc = np.array(texcoords, dtype=np.float32).reshape(-1, 2)

        self.set_geometry(va, na, ta)
        self.set_vertex_colors(color_float_to_ubyte(ca))
        self.texture_coordinates = tc

    _save_attrs = (
        "vertices",
        "normals",
        "triangles",
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
        for attr in self._save_attrs:
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

        geom_attrs = ("vertices", "normals", "triangles")
        self.set_geometry(data["vertices"], data["normals"], data["triangles"])
        for attr in PaperChainModel._save_attrs:
            if attr not in geom_attrs:
                setattr(self, attr, data[attr])
