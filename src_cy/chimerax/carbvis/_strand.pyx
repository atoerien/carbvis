# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language=c++
# cython: boundscheck=False, cdivision=True, language_level=3, wraparound=False

import numpy as np

from .utils import color_float_to_ubyte

from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref
from libc.math cimport acos, cos, pi, sin
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.vector cimport vector

cimport numpy as np
from chimerax.arrays.pythonarray cimport c_array_to_python

from ._carbs cimport atom_coord, atoms_coords, ring_get_centroid_and_normal
from ._utils cimport *

ctypedef PyObject *RingSphereMapKey
ctypedef (vector[int], vector[Color]) RingSphereMapT
ctypedef map[RingSphereMapKey, RingSphereMapT] RingSphereMap

cdef inline void ring_sphere_add(
    RingSphereMap &ring_spheres,
    PyObject *ring,
    int offset,
    Color color,
):
    it = ring_spheres.find(ring)
    if it == ring_spheres.end():
        v = pair[RingSphereMapKey, RingSphereMapT](ring, (vector[int](), vector[Color]()))
        p = ring_spheres.insert(v)
        it = p.first

    obj = &deref(it).second
    deref(obj)[0].push_back(offset)
    deref(obj)[1].push_back(color)

cdef void draw_tube(
    vector[Vec] &vertices,
    vector[Vec] &normals,
    vector[(int, int, int)] &triangles,
    vector[Color] &vcolors,
    RingSphereMap &ring_spheres,
    object linkage,
    Color color_a,
    Color color_b,
    double radius,
    bint candy_cane,
    int segment_subdivisions,
    int circle_subdivisions,
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

    # start the spline away from the ring centroids
    # will extend the tubes to meet the centroids later
    # this ensures the tube is not twisting
    # when it intersects with the spheres
    start_spline = vec_scaled_add(start_centroid, radius, start_tangent)
    end_spline = vec_scaled_sub(end_centroid, radius, end_tangent)

    spline_delta = vec_sub(start_spline, end_spline)
    cdef double spline_interval = vec_norm(spline_delta)

    cdef double tmp = 1.0 / spline_interval

    spline_a = Vec(0, 0, 0)
    spline_a = vec_scaled_add(spline_a, 2 * tmp, spline_delta)
    spline_a = vec_add(spline_a, start_tangent)
    spline_a = vec_add(spline_a, end_tangent)
    spline_a = vec_scale(tmp * tmp, spline_a)

    spline_b = Vec(0, 0, 0)
    spline_b = vec_scaled_sub(spline_b, 3 * tmp, spline_delta)
    spline_b = vec_scaled_sub(spline_b, 2, start_tangent)
    spline_b = vec_sub(spline_b, end_tangent)
    spline_b = vec_scale(tmp, spline_b)

    spline_c = start_tangent
    spline_d = start_spline

    cdef int n = segment_subdivisions + 3

    spath = vector[Vec](n)
    stan = vector[Vec](n)

    # start and end the path at the centroids
    spath[0] = start_centroid
    stan[0] = start_tangent
    spath[n - 1] = end_centroid
    stan[n - 1] = end_tangent

    cdef int i
    cdef double t = 0
    cdef double t_inc = spline_interval / (n - 3)

    for i in range(1, n - 1):
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

    cdef double correction_angle, rot_angle
    cdef double x, x_inc

    if candy_cane:
        p_frame = &frames[n - 2]

        end_right = vec_cross(end_tangent, end_normal)
        end_right = vec_normalize(end_right)

        correction_angle = acos(vec_dot(end_right, p_frame.right))
        if vec_dot(vec_cross(end_right, p_frame.right), end_tangent) > 0:
            correction_angle *= -1

        x = 0
        x_inc = 1.0 / (n - 3)

        # rotate frames about frame.forward (twist)
        # don't twist the first and last frames
        for i in range(1, n - 1):
            tmp = sin(pi * x / 2)
            rot_angle = correction_angle * tmp * tmp
            x += x_inc

            p_frame = &frames[i]
            p_frame.up = vec_rotate(p_frame.up, p_frame.forward, rot_angle)
            p_frame.right = vec_rotate(p_frame.right, p_frame.forward, rot_angle)

        # twist last frame the full angle to line up with the end ring
        p_frame = &frames[n - 1]
        p_frame.up = vec_rotate(p_frame.up, p_frame.forward, correction_angle)
        p_frame.right = vec_rotate(p_frame.right, p_frame.forward, correction_angle)

    cdef int triangle_offset = vertices.size()
    cdef int j
    cdef int offset, offset_next, k0, k1, k2, k3
    cdef double a
    cdef double a_inc = 2.0 * pi / circle_subdivisions

    for i in range(n):
        p_frame = &frames[i]

        # align the middle of the first stripe with frame.up
        a = -a_inc / 2

        for _ in range(circle_subdivisions):
            norm = vec_sub(vec_scale(cos(a), p_frame.up), vec_scale(sin(a), p_frame.right))
            vert = vec_scaled_add(p_frame.origin, radius, norm)
            a += a_inc

            vertices.push_back(vert)
            normals.push_back(norm)
            vcolors.push_back(
                # color_a if j > 1 else (0.0, 0.0, 0.0)
                color_a
            )

            if candy_cane:
                vertices.push_back(vert)
                normals.push_back(norm)
                vcolors.push_back(color_b)

        if i == n - 1:
            continue  # no normal triangles for the last frame

        if candy_cane:
            offset = triangle_offset + (2 * i) * circle_subdivisions
            offset_next = offset + 2 * circle_subdivisions
            for j in range(circle_subdivisions):
                k0 = offset + 2 * j
                k1 = offset + 2 * ((j + 1) % circle_subdivisions)
                k2 = offset_next + 2 * j
                k3 = offset_next + 2 * ((j + 1) % circle_subdivisions)

                if j % 2 == 1:
                    k0 += 1
                    k1 += 1
                    k2 += 1
                    k3 += 1

                triangles.push_back((k0, k2, k1))
                triangles.push_back((k1, k2, k3))
        else:
            offset = triangle_offset + i * circle_subdivisions
            offset_next = offset + circle_subdivisions
            for j in range(circle_subdivisions):
                k0 = offset + j
                k1 = offset + (j + 1) % circle_subdivisions
                k2 = offset_next + j
                k3 = offset_next + (j + 1) % circle_subdivisions
                triangles.push_back((k0, k2, k1))
                triangles.push_back((k1, k2, k3))

    if candy_cane:
        offset = triangle_offset
        p_ring = <PyObject *>start_ring
        ring_sphere_add(ring_spheres, p_ring, offset, color_a)

        offset = triangle_offset + (2 * (n - 1)) * circle_subdivisions
        p_ring = <PyObject *>end_ring
        ring_sphere_add(ring_spheres, p_ring, offset, color_a)
    else:
        offset = triangle_offset
        p_ring = <PyObject *>start_ring
        ring_sphere_add(ring_spheres, p_ring, offset, color_a)

        offset = triangle_offset + (n - 1) * circle_subdivisions
        p_ring = <PyObject *>end_ring
        ring_sphere_add(ring_spheres, p_ring, offset, color_a)

cdef void draw_sphere(
    vector[Vec] &vertices,
    vector[Vec] &normals,
    vector[(int, int, int)] &triangles,
    vector[Color] &vcolors,
    object ring,
    Color color_a,
    Color color_b,
    double radius,
    bint candy_cane,
    int circle_subdivisions,
):
    coords = atoms_coords(ring.atoms)
    origin, up = ring_get_centroid_and_normal(
        coords.data(), coords.size()
    )

    norm_top = Vec(0, 0, 1)
    norm_bot = Vec(0, 0, -1)

    # align top vector of sphere to up

    rot_axis = vec_cross(norm_top, up)
    cdef double axis_norm = vec_norm(rot_axis)
    cdef double rot_angle = 0

    # skip rotation if we're already aligned
    if axis_norm > 1e-4:
        rot_angle = acos(
            # float shenanigans
            clip(vec_dot(norm_top, up), -1.0, 1.0)
        )

        norm_top = vec_rotate(norm_top, rot_axis, rot_angle)
        norm_bot = vec_rotate(norm_bot, rot_axis, rot_angle)

    cdef int n_theta = circle_subdivisions // 2
    cdef double theta_inc = pi / n_theta
    cdef int n_phi = 128
    cdef double phi_inc = 2 * pi / n_phi

    cdef int triangle_offset = vertices.size()
    cdef int offset, offset_next, k0, k1, k2, k3

    cdef double phi
    cdef double theta = theta_inc / 2

    # use the z-component of the norm of the first latitude line
    # to scale the radius for the poles, so that the top and bottom
    # "faces" of the sphere are flat, to align with the tubes
    # HACK: multiply by 0.99 to try avoid z-fighting
    pole_radius = 0.99 * radius * cos(theta)

    vertices.push_back(vec_scaled_add(origin, pole_radius, norm_top))
    normals.push_back(norm_top)
    vcolors.push_back(color_a)

    cdef int i, j, next_j

    if candy_cane:
        offset = triangle_offset + 1
        k0 = triangle_offset
        for i in range(n_phi):
            k1 = offset + 2 * i
            k2 = offset + 2 * ((i + 1) % n_phi)
            triangles.push_back((k0, k1, k2))
    else:
        offset = triangle_offset + 1
        k0 = triangle_offset
        for i in range(n_phi):
            k1 = offset + i
            k2 = offset + (i + 1) % n_phi
            triangles.push_back((k0, k1, k2))

    for i in range(n_theta):
        if candy_cane:
            offset = triangle_offset + 1 + (2 * i) * n_phi
            offset_next = offset + 2 * n_phi
        else:
            offset = triangle_offset + 1 + i * n_phi
            offset_next = offset + n_phi

        phi = 0
        for j in range(n_phi):
            next_j = (j + 1) % n_phi

            norm = Vec(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta),
            )
            norm = vec_rotate(norm, rot_axis, rot_angle)
            vert = vec_scaled_add(origin, radius, norm)

            vertices.push_back(vert)
            normals.push_back(norm)
            vcolors.push_back(color_a)

            if candy_cane:
                vertices.push_back(vert)
                normals.push_back(norm)
                vcolors.push_back(color_b)

            # no triangles for the last row
            if i == n_theta - 1:
                phi += phi_inc
                continue

            if candy_cane:
                k0 = offset + 2 * j
                k1 = offset + 2 * next_j
                k2 = offset_next + 2 * j
                k3 = offset_next + 2 * next_j

                if i % 2 == 0:
                    k0 += 1
                    k1 += 1
                    k2 += 1
                    k3 += 1

                triangles.push_back((k0, k2, k1))
                triangles.push_back((k1, k2, k3))
            else:
                k0 = offset + j
                k1 = offset + next_j
                k2 = offset_next + j
                k3 = offset_next + next_j
                triangles.push_back((k0, k2, k1))
                triangles.push_back((k1, k2, k3))

            phi += phi_inc

        theta += theta_inc

    vertices.push_back(vec_scaled_add(origin, pole_radius, norm_bot))
    normals.push_back(norm_bot)
    vcolors.push_back(color_a)

    if candy_cane:
        offset = triangle_offset + 1 + 2 * (n_theta - 1) * n_phi
        k0 = vertices.size() - 1
        for i in range(n_phi):
            k1 = offset + 2 * i
            k2 = offset + 2 * ((i + 1) % n_phi)
            triangles.push_back((k0, k2, k1))
    else:
        offset = triangle_offset + 1 + (n_theta - 1) * n_phi
        k0 = vertices.size() - 1
        for i in range(n_phi):
            k1 = offset + i
            k2 = offset + (i + 1) % n_phi
            triangles.push_back((k0, k2, k1))


def update_graphics(object model not None):
    cdef double radius = model.radius
    colormap = model.dihedral_colormap
    cdef bint candy_cane = model.candy_cane
    cdef double sphere_radius = model.sphere_radius
    sphere_colormap = model.sphere_colormap

    cdef vector[Vec] vertices
    cdef vector[Vec] normals
    cdef vector[(int, int, int)] triangles
    cdef vector[Color] vcolors
    cdef vector[(int, int)] r2t
    cdef vector[int] t2r
    cdef vector[(int, int)] l2t
    cdef vector[int] t2l

    cdef int segment_subdivisions = 20
    cdef int circle_subdivisions = 16

    color_b = Color(1, 1, 1, 1)

    cdef RingSphereMap ring_spheres

    cdef int i
    cdef np.ndarray color_np
    cdef list linkages = model.linkages

    for i in range(len(linkages)):
        link = linkages[i]
        color_np = colormap(link)
        color_a = color_from_array(<float *>color_np.data)

        tri_before = triangles.size()
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
        tri_after = triangles.size()

        l2t.push_back((tri_before, tri_after))
        for _ in range(tri_before, tri_after):
            t2l.push_back(i)
            t2r.push_back(-1)

    sphere_radius = max(sphere_radius, radius)

    cdef bint sphere_candy_cane
    if sphere_radius == radius:
        sphere_candy_cane = candy_cane
        sphere_colormap = None
    else:
        sphere_candy_cane = False

    cdef int offset, j

    cdef list rings = model.rings
    for i in range(len(rings)):
        ring = rings[i]
        offset_list, color_list = ring_spheres[<PyObject *>ring]

        if sphere_colormap is not None:
            color_np = sphere_colormap(ring)
            color_a = color_from_array(<float *>color_np.data)
        else:
            # get average color
            color_a = Color(0, 0, 0, 0)
            for c in color_list:
                color_a = color_add(color_a, c)
            color_a = color_scale(1.0 / color_list.size(), color_a)

            if sphere_radius == radius:
                # overwrite vcolors for each adjacent frame
                # this blends colors through the ring
                for offset in offset_list:
                    if candy_cane:
                        for j in range(0, 2 * circle_subdivisions, 2):
                            vcolors[offset + j] = color_a
                    else:
                        for j in range(circle_subdivisions):
                            vcolors[offset + j] = color_a

        tri_before = triangles.size()
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
