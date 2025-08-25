from __future__ import annotations

import numpy as np
from chimerax.atomic import Structure
from chimerax.atomic.changes import Changes
from chimerax.core.session import Session
from chimerax.graphics import Texture

from .carbs import find_rings, paperchain_colormap
from .model import CarbVisModel
from .utils import FloatArray, Frame, UByteArray, color_float_to_ubyte, time


def make_texture(
    formula="stripes",
    size=1024,
    period=64,
    duty=0.5,
    on_color=255,
    off_color=0,
) -> UByteArray:
    """
    Generate a procedural texture pattern.

    Parameters:
        formula: one of
            'stripes', 'grid', 'diamond',
            'rings', 'waves'
        size: texture size in pixels (square)
        period: pixel distance between repeating features
        duty: fraction of each period that is 'off' (0.0-1.0)
        on_color: int 0-255 or (R,G,B) tuple for 'on' pixels
        off_color: int 0-255 or (R,G,B) tuple for 'off' pixels
    """

    # allow int or tuple color
    def to_rgb(c):
        if isinstance(c, (tuple, list, np.ndarray)):
            return np.array(c, dtype=np.uint8)
        else:
            return np.array((c, c, c), dtype=np.uint8)

    on_rgb = to_rgb(on_color)
    off_rgb = to_rgb(off_color)

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

    tex = np.where(mask[..., None], off_rgb, on_rgb).astype(np.uint8)
    return tex


def ring_tex(ring_coords: FloatArray, frame: Frame) -> FloatArray:
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


class PaperChainModel(CarbVisModel):
    def __init__(
        self,
        session: Session,
        structure: Structure,
        name: str | None = None,
        *,
        update: bool,
        bipyramid_height: float,
        max_ring_size: int,
        tex_formula: str | None,
    ):
        if name is None:
            name = f"{structure.name} PaperChain"
        super().__init__(session, structure, name, update=update)

        self.bipyramid_height = bipyramid_height
        self.max_ring_size = max_ring_size

        self.texture: Texture = Texture(linear_interpolation=True)
        self.tex_formula = tex_formula
        self._update_texture()

    def update_params(
        self,
        *,
        update: bool,
        bipyramid_height: float,
        max_ring_size: int,
        tex_formula: str | None,
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

        # TODO: make more texture params configurable
        update_texture = False
        if tex_formula != self.tex_formula:
            self.tex_formula = tex_formula
            update_texture = True

        if clear_geometry:
            self._clear_geometry()
        if update_texture:
            self._update_texture()

    def _update_texture(self):
        if self.tex_formula is not None:
            tex = make_texture(
                formula=self.tex_formula,
                period=128,
                duty=0.5,
                on_color=255,
                off_color=128,
            )
        else:
            tex = np.full((16, 16, 3), 255, dtype=np.uint8)
        self.texture.reload_texture(tex)

    def _do_auto_update(self, changes: Changes):
        super()._do_auto_update(changes)

        # TODO: does this work for new bonds/atoms?
        if self._coordinates_changed(changes):
            self._recalculate_graphics()

    @time
    def _calc_graphics(self):
        bipyramid_height = self.bipyramid_height

        # NOTE: should point to a white part of the texture
        # TODO: make this better
        TOP_TEXTURE_COORD = (0.49, 0.49)

        vertices = []
        normals = []
        triangles = []
        vcolors = []
        texcoords = []

        rings = find_rings(self.structure, self.max_ring_size)

        for ring in rings:
            n = len(ring.atoms)  # the number of atoms in the current ring

            pucker = ring.calc_pucker_amplitude()
            color = paperchain_colormap(pucker)

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
            texcoords.append(TOP_TEXTURE_COORD)

            vertices.append(bottom)
            normals.append(-frame.up)
            vcolors.append(color)
            texcoords.append((0.5, 0.5))

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

                vertices.append(curvec)
                normals.append(normtop)
                vcolors.append(color)
                texcoords.append(TOP_TEXTURE_COORD)

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
                texcoords.append(tex[i])

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
