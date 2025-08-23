from __future__ import annotations

import numpy as np
from chimerax.atomic import Structure
from chimerax.atomic.changes import Changes
from chimerax.core.session import Session
from chimerax.graphics import Texture

from .carbs import CarbRing, find_rings
from .model import CarbVisModel
from .utils import FloatArray, Frame, color_float_to_ubyte, time


def cremer_pople_ring_pucker(ring: CarbRing) -> float:
    displ = atom_displ_from_mean_plane(ring.coords)

    q = np.sqrt(np.sum(displ**2))
    return min(q, 2.0)  # truncate amplitude at 2


def atom_displ_from_mean_plane(coords: FloatArray) -> FloatArray:
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


def colormap(pucker_sum: float) -> FloatArray:
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

    return rgb


def make_texture(
    formula="stripes",
    size=1024,
    period=64,
    duty=0.5,
    on_color=255,
    off_color=0,
):
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
        mask = ((x % period) < (period * duty)) | ((y % period) < (period * duty))
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
        update=True,
    ):
        if name is None:
            name = f"{structure.name} PaperChain"
        super().__init__(session, structure, name, update)

        tex = make_texture(
            formula="waves",
            period=128,
            duty=0.5,
            on_color=255,
            off_color=128,
        )
        self.texture = Texture(tex)

    def _do_auto_update(self, changes: Changes):
        super()._do_auto_update(changes)

        # TODO: does this work for new bonds/atoms?
        if self._coordinates_changed(changes):
            self._recalculate_graphics()

    @time
    def _calc_graphics(self):
        # FIXME: not hardcode
        maxringsize = 10
        bipyramid_height = 0.5
        have_texture = self.texture is not None

        # NOTE: should point to a white part of the texture
        # TODO: make this better
        TOP_TEXTURE_COORD = (0.49, 0.49)

        vertices = []
        normals = []
        triangles = []
        vcolors = []
        texcoords = []

        rings = find_rings(self.structure, maxringsize)

        for ring in rings:
            n = len(ring.atoms)  # the number of atoms in the current ring

            pucker_sum = cremer_pople_ring_pucker(ring)
            color = colormap(pucker_sum)

            ring_coords = ring.coords
            frame = ring.get_frame()

            x = 0.5 * bipyramid_height * frame.up
            top = frame.origin + x
            bottom = frame.origin - x

            triangle_offset = len(vertices)

            vertices.append(top)
            normals.append(frame.up)
            vcolors.append(color)

            vertices.append(bottom)
            normals.append(-frame.up)
            vcolors.append(color)

            if have_texture:
                tex = ring_tex(ring_coords, frame)

                texcoords.append(TOP_TEXTURE_COORD)
                texcoords.append((0.5, 0.5))
            else:
                tex = None

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
                if have_texture:
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
                if have_texture:
                    assert tex is not None
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
