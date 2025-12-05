# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language=c++
# cython: boundscheck=False, cdivision=True, language_level=3, wraparound=False

import numpy as np

from .utils import color_float_to_ubyte

from libc.math cimport sqrt
from libcpp.vector cimport vector

cimport numpy as np
from chimerax.arrays.pythonarray cimport c_array_to_python

from ._carbs cimport atoms_coords, ring_get_frame
from ._utils cimport *

cdef vector[(float, float)] ring_tex(vector[Vec] &coords, CyFrame &frame):
    """Calculate ring texture coordinates."""
    cdef vector[(float, float)] tex

    cdef int n = coords.size()
    tex.reserve(n)

    # project coordinates down into the (forward, right) plane

    cdef int i
    cdef double x, y, r
    cdef double r_max = 0

    for i in range(n):
        p = coords[i]

        dr = vec_sub(p, frame.origin)
        dr_plane = vec_sub(dr, vec_proj(dr, frame.up))
        x = vec_dot(dr_plane, frame.forward)
        y = vec_dot(dr_plane, frame.right)
        tex.push_back((x, y))
        r = sqrt(x * x + y * y)
        if r > r_max:
            r_max = r

    # scale and translate to [0, 1] texture space
    if r_max == 0:
        # degenerate: all points at centre
        for i in range(n):
            tex[i] = (0.5, 0.5)
    else:
        for i in range(n):
            x, y = tex[i]
            tex[i] = (
                x / (2.0 * r_max) + 0.5,
                y / (2.0 * r_max) + 0.5,
            )

    return tex

cdef void draw_ring(
    vector[Vec] &vertices,
    vector[Vec] &normals,
    vector[(int, int, int)] &triangles,
    vector[CyColor] &vcolors,
    vector[(float, float)] &texcoords,
    object ring,
    CyColor color,
    float bipyramid_height,
):
    coords = atoms_coords(ring.atoms)
    cdef int n = coords.size()

    frame = ring_get_frame(coords.data(), n)
    tex = ring_tex(coords, frame)

    down = vec_negate(frame.up)

    x = vec_scale(0.5 * bipyramid_height, frame.up)
    top = vec_add(frame.origin, x)
    bottom = vec_sub(frame.origin, x)

    cdef int triangle_offset = vertices.size()

    vertices.push_back(top)
    normals.push_back(frame.up)
    vcolors.push_back(color)
    texcoords.push_back((0.5, 0.25))

    vertices.push_back(bottom)
    normals.push_back(down)
    vcolors.push_back(color)
    texcoords.push_back((0.5, 0.75))

    cdef int i, next_i
    cdef (float, float) textop, texbot

    for i in range(n):
        next_i = (i + 1) % n

        curvec = coords[i]
        nextvec = coords[next_i]
        delta = vec_sub(nextvec, curvec)

        normtop = vec_cross(delta, vec_sub(curvec, top))
        normtop = vec_normalize(normtop)

        # reverse normal direction for bottom half
        normbot = vec_cross(vec_sub(curvec, bottom), delta)
        normbot = vec_normalize(normbot)

        tex_x, tex_y = tex[i]
        textop = (tex_x, tex_y / 2)
        texbot = (tex_x, tex_y / 2 + 0.5)

        vertices.push_back(curvec)
        normals.push_back(normtop)
        vcolors.push_back(color)
        texcoords.push_back(textop)

        triangles.push_back(
            (
                triangle_offset + 2 + (2 * next_i),
                triangle_offset + 2 + (2 * i),
                triangle_offset,
            )
        )

        vertices.push_back(curvec)
        normals.push_back(normbot)
        vcolors.push_back(color)
        texcoords.push_back(texbot)

        # order different to keep anticlockwise winding
        triangles.push_back(
            (
                triangle_offset + 3 + (2 * i),
                triangle_offset + 3 + (2 * next_i),
                triangle_offset + 1,
            )
        )

def update_graphics(object model not None):
    cdef double bipyramid_height = model.bipyramid_height

    cdef np.ndarray color_np
    cmap = model.cmap
    if not callable(cmap):
        color_np = cmap.rgba
        color = color_from_array(<float *>color_np.data)
        cmap = None

    cdef vector[Vec] vertices
    cdef vector[Vec] normals
    cdef vector[(int, int, int)] triangles
    cdef vector[CyColor] vcolors
    cdef vector[(float, float)] texcoords
    cdef vector[(int, int)] r2t
    cdef vector[int] t2r

    cdef int i
    cdef list rings = model.rings

    for i in range(len(rings)):
        ring = rings[i]

        if cmap is not None:
            color_np = cmap(ring).rgba
            color = color_from_array(<float *>color_np.data)

        tri_before = triangles.size()
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
        tri_after = triangles.size()

        r2t.push_back((tri_before, tri_after))
        for _ in range(tri_before, tri_after):
            t2r.push_back(i)

    va = c_array_to_python(<double *>vertices.data(), vertices.size(), 3)
    na = c_array_to_python(<double *>normals.data(), normals.size(), 3)
    ta = c_array_to_python(<int *>triangles.data(), triangles.size(), 3)
    ca = c_array_to_python(<float *>vcolors.data(), vcolors.size(), 4)
    tc = c_array_to_python(<float *>texcoords.data(), texcoords.size(), 2)
    r2ta = c_array_to_python(<int *>r2t.data(), r2t.size(), 2)
    t2ra = c_array_to_python(t2r)

    model.set_geometry(va, na, ta)
    model.ring_to_triangle = r2ta
    model.triangle_to_ring = t2ra
    model.set_vertex_colors(color_float_to_ubyte(ca))
    model.texture_coordinates = tc
    model.update_selection()
