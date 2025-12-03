# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language=c++
# cython: boundscheck=False, cdivision=True, language_level=3, wraparound=False

import numpy as np

from .utils import color_float_to_ubyte

from cpython.ref cimport PyObject
from libc.math cimport acos, pi, pow, sin
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.vector cimport vector

cimport numpy as np
from chimerax.arrays.pythonarray cimport c_array_to_python

from ._carbs cimport (
    atom_coord,
    atoms_coords,
    ring_get_centroid_and_normal,
    ring_get_frame,
)
from ._utils cimport *

ctypedef PyObject *RingColorMapKey
ctypedef Color RingColorMapT
ctypedef map[RingColorMapKey, RingColorMapT] RingColorMap

cdef void draw_ribbon(
    vector[Vec] &vertices,
    vector[Vec] &normals,
    vector[(int, int, int)] &triangles,
    vector[Color] &vcolors,
    object linkage,
    Color color_top,
    Color color_bottom,
    bint start_end_centroid,
    int rib_steps,
    double rib_width,
    double rib_height,
    bint gum_twist,
):
    start_ring = linkage.start_ring
    start_ring_coords = atoms_coords(start_ring.atoms)
    start_centroid, start_normal = ring_get_centroid_and_normal(
        start_ring_coords.data(), start_ring_coords.size()
    )

    end_ring = linkage.end_ring
    end_ring_coords = atoms_coords(end_ring.atoms)
    end_centroid, end_normal = ring_get_centroid_and_normal(
        end_ring_coords.data(), end_ring_coords.size()
    )

    start_coord = atom_coord(linkage.atoms[0])
    end_coord = atom_coord(linkage.atoms[len(linkage.atoms) - 1])

    start_tangent = vec_sub(start_coord, start_centroid)
    cdef double dot = vec_dot(start_tangent, start_normal)
    start_tangent = vec_scaled_sub(start_tangent, dot, start_normal)
    start_tangent = vec_normalize(start_tangent)

    end_tangent = vec_sub(end_centroid, end_coord)  # reversed direction for end tangent
    dot = vec_dot(end_tangent, end_normal)
    end_tangent = vec_scaled_sub(end_tangent, dot, end_normal)
    end_tangent = vec_normalize(end_tangent)

    if start_end_centroid:
        # move start and end points towards their centroid
        # and then onto the plane formed by the centroid and
        # the ring normal
        start_rib = vec_scale(0.5, vec_add(start_centroid, start_coord))
        dot = vec_dot(vec_sub(start_rib, start_centroid), start_normal)
        start_rib = vec_scaled_sub(start_rib, dot, start_normal)

        end_rib = vec_scale(0.5, vec_add(end_centroid, end_coord))
        dot = vec_dot(vec_sub(end_rib, end_centroid), end_normal)
        end_rib = vec_scaled_sub(end_rib, dot, end_normal)
    else:
        start_rib = start_coord
        end_rib = end_coord

    rib_delta = vec_sub(start_rib, end_rib)
    cdef double rib_interval = vec_norm(rib_delta)

    cdef double tmp = 1.0 / rib_interval

    spline_a = Vec(0, 0, 0)
    spline_a = vec_scaled_add(spline_a, 2 * tmp, rib_delta)
    spline_a = vec_add(spline_a, start_tangent)
    spline_a = vec_add(spline_a, end_tangent)
    spline_a = vec_scale(tmp * tmp, spline_a)

    spline_b = Vec(0, 0, 0)
    spline_b = vec_scaled_sub(spline_b, 3 * tmp, rib_delta)
    spline_b = vec_scaled_sub(spline_b, 2, start_tangent)
    spline_b = vec_sub(spline_b, end_tangent)
    spline_b = vec_scale(tmp, spline_b)

    spline_c = start_tangent
    spline_d = start_rib

    cdef int i
    cdef int n = rib_steps + 1
    cdef double t = 0
    cdef double t_inc = rib_interval / (n - 1)

    cdef int n_spline = n

    cdef vector[Vec] spath, stan

    if start_end_centroid:
        # extend the path to meet the centroids

        n += 2
        spath.resize(n)
        stan.resize(n)

        spath[0] = start_centroid
        stan[0] = start_tangent

        for i in range(1, n - 1):
            spath[i], stan[i] = spline(spline_a, spline_b, spline_c, spline_d, t)
            t += t_inc

        spath[n - 1] = end_centroid
        stan[n - 1] = end_tangent
    else:
        spath.resize(n)
        stan.resize(n)

        for i in range(n):
            spath[i], stan[i] = spline(spline_a, spline_b, spline_c, spline_d, t)
            t += t_inc

    frames = vector[CyFrame](n)

    # Initial frame
    point = spath[0]
    tangent = stan[0]
    up = start_normal
    up = vec_scaled_sub(up, vec_dot(up, tangent), tangent)
    if vec_allclose(up, 0.0):
        # fallback: pick a world axis
        if abs(tangent.x) < 0.9:
            up = Vec(1.0, 0.0, 0.0)
        else:
            up = Vec(0.0, 1.0, 0.0)
        up = vec_scaled_sub(up, vec_dot(up, tangent), tangent)
    up = vec_normalize(up)

    # tangent and forward are length 1, no need to norm
    right = vec_cross(tangent, up)

    frames[0] = CyFrame(point, up, tangent, right)

    for i in range(1, n):
        point = spath[i]
        tangent = stan[i]

        # copy previous frame
        frame = frames[i - 1]
        frame.origin = point

        # align frame with tangent
        frame_align(frame, tangent)

        frames[i] = frame

    cdef CyFrame *spline_frames
    if start_end_centroid:
        spline_frames = &frames[1]
    else:
        spline_frames = &frames[0]

    p_frame = &spline_frames[n_spline - 1]

    end_right = vec_cross(end_tangent, end_normal)
    end_right = vec_normalize(end_right)

    cdef double correction_angle = acos(vec_dot(end_right, p_frame.right))
    if vec_dot(vec_cross(end_right, p_frame.right), end_tangent) > 0:
        correction_angle *= -1

    if start_end_centroid:
        # twist last frame the full angle to line up with the end ring
        p_frame = &frames[n - 1]
        p_frame.up = vec_rotate(p_frame.up, p_frame.forward, correction_angle)
        p_frame.right = vec_rotate(p_frame.right, p_frame.forward, correction_angle)

    cdef double rot_angle
    cdef double x = 0
    cdef double x_inc = 1.0 / (n_spline - 1)

    # rotate frames about frame.forward (twist)
    for i in range(n_spline):
        p_frame = &spline_frames[i]

        if gum_twist:
            tmp = sin(pi * x / 2)
            rot_angle = correction_angle * tmp * tmp
        else:
            rot_angle = correction_angle * x
        x += x_inc

        p_frame.up = vec_rotate(p_frame.up, p_frame.forward, rot_angle)
        p_frame.right = vec_rotate(p_frame.right, p_frame.forward, rot_angle)

    cdef int triangle_offset = vertices.size()
    cdef int offset, offset_next
    cdef double width = rib_width
    cdef double wave, width_factor

    x = 0
    for i in range(n):
        p_frame = &frames[i]

        if gum_twist:
            if start_end_centroid and (i == 0 or i == n - 1):
                width = rib_width
            else:
                wave = pow(sin(pi * x), 4)
                width_factor = 0.75 * abs(correction_angle) / pi
                width = rib_width * (1 - width_factor * wave)
                x += x_inc

        # vertices (of this frame's rectangle)

        # top right (index +0)
        vert = vec_scaled_add(p_frame.origin, rib_height, p_frame.up)
        vert = vec_scaled_add(vert, width, p_frame.right)
        vertices.push_back(vert)
        norm = vec_add(p_frame.up, p_frame.right)
        norm = vec_normalize(norm)
        normals.push_back(norm)
        vcolors.push_back(color_top)

        # bottom right (index +1)
        vert = vec_scaled_sub(p_frame.origin, rib_height, p_frame.up)
        vert = vec_scaled_add(vert, width, p_frame.right)
        vertices.push_back(vert)
        norm = vec_sub(p_frame.right, p_frame.up)
        norm = vec_normalize(norm)
        normals.push_back(norm)
        vcolors.push_back(color_bottom)

        # bottom left (index +2)
        vert = vec_scaled_sub(p_frame.origin, rib_height, p_frame.up)
        vert = vec_scaled_sub(vert, width, p_frame.right)
        vertices.push_back(vert)
        norm = vec_sub(vec_negate(p_frame.right), p_frame.up)
        norm = vec_normalize(norm)
        normals.push_back(norm)
        vcolors.push_back(color_bottom)

        # top left (index +3)
        vert = vec_scaled_add(p_frame.origin, rib_height, p_frame.up)
        vert = vec_scaled_sub(vert, width, p_frame.right)
        vertices.push_back(vert)
        norm = vec_sub(p_frame.up, p_frame.right)
        norm = vec_normalize(norm)
        normals.push_back(norm)
        vcolors.push_back(color_top)

        # no triangles for the last frame
        if i == n - 1:
            continue

        offset = triangle_offset + i * 4
        offset_next = offset + 4

        # top 1
        triangles.push_back(
            (
                offset,  # current, top right
                offset_next,  # next, top right
                offset + 3,  # current, top left
            )
        )

        # top 2
        triangles.push_back(
            (
                offset_next,  # next, top right
                offset_next + 3,  # next, top left
                offset + 3,  # current, top left
            )
        )

        # bottom 1
        triangles.push_back(
            (
                offset + 1,  # current, bottom right
                offset + 2,  # current, bottom left
                offset_next + 1,  # next, bottom right
            )
        )

        # bottom 2
        triangles.push_back(
            (
                offset_next + 1,  # next, bottom right
                offset + 2,  # current, bottom left
                offset_next + 2,  # next, bottom left
            )
        )

        # right 1
        triangles.push_back(
            (
                offset,  # current, top right
                offset + 1,  # current, bottom right
                offset_next,  # next, top right
            )
        )

        # right 2
        triangles.push_back(
            (
                offset_next,  # next, top right
                offset + 1,  # current, bottom right
                offset_next + 1,  # next, bottom right
            )
        )

        # left 1
        triangles.push_back(
            (
                offset + 3,  # current, top left
                offset_next + 3,  # next, top left
                offset + 2,  # current, bottom left
            )
        )

        # left 2
        triangles.push_back(
            (
                offset_next + 3,  # next, top left
                offset_next + 2,  # next, bottom left
                offset + 2,  # current, bottom left
            )
        )

    if not start_end_centroid:
        # TODO: fix weird lighting, probably need to dupe edge vertices

        # Draw start and end end caps
        # Start end caps
        offset = triangle_offset

        triangles.push_back(
            (
                offset + 3,  # top left
                offset,  # top right
                offset + 1,  # bottom right
            )
        )

        triangles.push_back(
            (
                offset + 2,  # bottom left
                offset + 1,  # bottom right
                offset + 3,  # top left
            )
        )

        # End end caps
        offset = triangle_offset + rib_steps * 4

        triangles.push_back(
            (
                offset + 3,  # top left
                offset,  # top right
                offset + 1,  # bottom right
            )
        )

        triangles.push_back(
            (
                offset + 2,  # bottom left
                offset + 1,  # bottom right
                offset + 3,  # top left
            )
        )

cdef extern from *:
    """
    const int POLYGON_N = 12;
    const double ROT_ANGLE = 2 * M_PI / POLYGON_N;
    """
    const int POLYGON_N
    const double ROT_ANGLE

cdef void draw_disk(
    vector[Vec] &vertices,
    vector[Vec] &normals,
    vector[(int, int, int)] &triangles,
    vector[Color] &vcolors,
    object ring,
    Color color_top,
    Color color_bottom,
    double rib_height,
    double rib_width,
):
    # TODO: fix weird lighting, probably need to dupe edge vertices
    # TODO: fix z-fighting?

    coords = atoms_coords(ring.atoms)
    frame = ring_get_frame(coords.data(), coords.size())

    cdef int triangle_offset = vertices.size()

    # top centroid vertex
    vert = vec_scaled_add(frame.origin, rib_height, frame.up)
    vertices.push_back(vert)
    norm = frame.up
    normals.push_back(norm)
    vcolors.push_back(color_top)

    # bottom centroid vertex
    vert = vec_scaled_sub(frame.origin, rib_height, frame.up)
    vertices.push_back(vert)
    norm = vec_negate(frame.up)
    normals.push_back(norm)
    vcolors.push_back(color_bottom)

    # edge vertices

    current_vec = frame.right

    cdef int i
    for i in range(POLYGON_N):
        # top hexagon vertex
        vert = vec_scaled_add(frame.origin, rib_height, frame.up)
        vert = vec_scaled_add(vert, rib_width, current_vec)
        vertices.push_back(vert)
        norm = vec_add(current_vec, frame.up)
        norm = vec_normalize(norm)
        normals.push_back(norm)
        vcolors.push_back(color_top)

        # bottom hexagon vertex
        vert = vec_scaled_sub(frame.origin, rib_height, frame.up)
        vert = vec_scaled_add(vert, rib_width, current_vec)
        vertices.push_back(vert)
        norm = vec_sub(current_vec, frame.up)
        norm = vec_normalize(norm)
        normals.push_back(norm)
        vcolors.push_back(color_bottom)

        if i == POLYGON_N - 1:
            break  # don't bother rotating the last time

        current_vec = vec_rotate(current_vec, frame.up, ROT_ANGLE)

    cdef int edge1_top, edge1_bottom, edge2_top, edge2_bottom

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
        triangles.push_back(
            (
                triangle_offset,
                edge1_top,
                edge2_top,
            )
        )

        # bottom
        triangles.push_back(
            (
                triangle_offset + 1,
                edge2_bottom,
                edge1_bottom,
            )
        )

        # outer edge
        triangles.push_back(
            (
                edge1_bottom,
                edge2_bottom,
                edge1_top,
            )
        )
        triangles.push_back(
            (
                edge1_top,
                edge2_bottom,
                edge2_top,
            )
        )

cdef inline void ring_color_add(
    RingColorMap &ring_colors,
    PyObject *ring,
    Color color,
):
    v = pair[RingColorMapKey, RingColorMapT](ring, color)
    ring_colors.insert(v)

def update_graphics(object model not None):
    cdef bint start_end_centroid = model.start_end_centroid
    cdef int rib_steps = model.rib_steps
    cdef double rib_width = model.rib_width
    cdef double rib_height = model.rib_height
    colormap = model.dihedral_colormap
    cdef bint gum_twist = model.gum_twist

    cdef vector[Vec] vertices
    cdef vector[Vec] normals
    cdef vector[(int, int, int)] triangles
    cdef vector[Color] vcolors
    cdef vector[(int, int)] r2t
    cdef vector[int] t2r
    cdef vector[(int, int)] l2t
    cdef vector[int] t2l

    color_top = Color(0.9, 0.9, 0.9, 1.0)
    color_bottom = Color(0.5, 0.5, 1.0, 1.0)

    cdef RingColorMap ring_colors

    cdef int i
    cdef np.ndarray color_np
    cdef list linkages = model.linkages

    for i in range(len(linkages)):
        link = linkages[i]

        if colormap is not None:
            color_np = colormap(link)
            color = color_from_array(<float *>color_np.data)
            color_top = color
            color_bottom = color

            if start_end_centroid:
                ring_color_add(ring_colors, <PyObject *>link.start_ring, color)
                ring_color_add(ring_colors, <PyObject *>link.end_ring, color)

        tri_before = triangles.size()
        draw_ribbon(
            vertices,
            normals,
            triangles,
            vcolors,
            link,
            color_top,
            color_bottom,
            start_end_centroid,
            rib_steps,
            rib_width,
            rib_height,
            gum_twist,
        )
        tri_after = triangles.size()

        l2t.push_back((tri_before, tri_after))
        for _ in range(tri_before, tri_after):
            t2l.push_back(i)
            t2r.push_back(-1)

    cdef list rings

    if start_end_centroid:
        rings = model.rings
        for i in range(len(rings)):
            ring = rings[i]

            if colormap is not None:
                color = ring_colors[<PyObject *>ring]
                color_top = color
                color_bottom = color

            tri_before = triangles.size()
            draw_disk(
                vertices,
                normals,
                triangles,
                vcolors,
                ring,
                color_top,
                color_bottom,
                rib_height,
                rib_width,
            )
            tri_after = triangles.size()

            r2t.push_back((tri_before, tri_after))
            for _ in range(tri_before, tri_after):
                t2r.push_back(i)
                t2l.push_back(-1)

    va = c_array_to_python(<double *>vertices.data(), vertices.size(), 3)
    na = c_array_to_python(<double *>normals.data(), normals.size(), 3)
    ta = c_array_to_python(<int *>triangles.data(), triangles.size(), 3)
    ca = c_array_to_python(<float *>vcolors.data(), vcolors.size(), 4)
    r2ta = c_array_to_python(<int *>r2t.data(), r2t.size(), 2)
    t2ra = c_array_to_python(t2r)
    l2ta = c_array_to_python(<int *>l2t.data(), l2t.size(), 2)
    t2la = c_array_to_python(t2l)

    model.set_geometry(va, na, ta)
    model.ring_to_triangle = r2ta
    model.triangle_to_ring = t2ra
    model.link_to_triangle = l2ta
    model.triangle_to_link = t2la
    model.set_vertex_colors(color_float_to_ubyte(ca))
    model.update_selection()
