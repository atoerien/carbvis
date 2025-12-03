# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language=c++
# cython: boundscheck=False, cdivision=True, language_level=3, wraparound=False

import numpy as np

from cython.operator cimport dereference as deref
from libc.math cimport cos, pi, sin, sqrt
from libcpp.vector cimport vector

from chimerax.atomic_lib.atomstruct cimport Atom

cimport numpy as np

from ._utils cimport *

cdef vector[Vec] atoms_coords(object atoms):
    cdef np.ndarray arr = atoms._pointer_array

    p = <Atom **>arr.data
    cdef int n = arr.shape[0]

    cdef vector[Vec] ret
    ret.reserve(n)

    cdef int i
    for i in range(n):
        coord = &p[i].coord()
        ret.push_back(
            Vec(deref(coord)[0], deref(coord)[1], deref(coord)[2])
        )

    return ret

cdef Vec ring_get_centroid(const Vec *coords, int n) noexcept:
    centroid = Vec(0, 0, 0)

    cdef int i
    for i in range(n):
        centroid = vec_add(centroid, coords[i])

    return vec_scale(1.0 / n, centroid)

def ring_get_centroid_py(object ring):
    coords = atoms_coords(ring.atoms)
    centroid = ring_get_centroid(coords.data(), coords.size())
    return vec_to_ndarray(centroid)

cdef (Vec, Vec) ring_get_centroid_and_normal(const Vec *coords, int n) noexcept:
    centroid = Vec(0, 0, 0)
    normal = Vec(0, 0, 0)

    cdef int i, next_i
    for i in range(n):
        next_i = (i + 1) % n

        curvec = coords[i]
        nextvec = coords[next_i]

        # update centroid
        centroid = vec_add(centroid, curvec)

        # update normal (this is Newell's method; see Carbohydra paper)
        normal.x += (curvec.y - nextvec.y) * (curvec.z + nextvec.z)
        normal.y += (curvec.z - nextvec.z) * (curvec.x + nextvec.x)
        normal.z += (curvec.x - nextvec.x) * (curvec.y + nextvec.y)

    centroid = vec_scale(1.0 / n, centroid)
    normal = vec_normalize(normal)

    # flip - the "up" of a carbohydrate ring is a LH normal
    # but Newell's method gives a RH normal
    normal = vec_negate(normal)

    return centroid, normal

def ring_get_centroid_and_normal_py(object ring):
    coords = atoms_coords(ring.atoms)
    centroid, normal = ring_get_centroid_and_normal(coords.data(), coords.size())
    return vec_to_ndarray(centroid), vec_to_ndarray(normal)

cdef CyFrame ring_get_frame(const Vec *coords, int n) noexcept:
    cdef CyFrame frame

    frame.origin, frame.up = ring_get_centroid_and_normal(coords, n)

    # use the first atom as forward, should not be parallel to up
    frame.forward = vec_sub(coords[0], frame.origin)
    cdef double dot = vec_dot(frame.forward, frame.up)
    frame.forward = vec_scaled_sub(frame.forward, dot, frame.up)

    if vec_allclose(frame.forward, 0.0):
        # fallback: pick a world axis
        if abs(frame.up.x) < 0.9:
            frame.forward = Vec(1.0, 0.0, 0.0)
        else:
            frame.forward = Vec(0.0, 1.0, 0.0)
        dot = vec_dot(frame.forward, frame.up)
        frame.forward = vec_scaled_sub(frame.forward, dot, frame.up)

    frame.forward = vec_normalize(frame.forward)

    # up and forward are length 1, no need to norm
    frame.right = vec_cross(frame.up, frame.forward)

    return frame

def ring_get_frame_py(object ring):
    coords = atoms_coords(ring.atoms)
    cdef Frame ret = Frame.__new__(Frame)
    ret.frame = ring_get_frame(coords.data(), coords.size())
    return ret

cdef double ring_calc_pucker_amplitude(const Vec *coords, int n) noexcept:
    centroid = ring_get_centroid(coords, n)

    # Calculate cartesian axes based on coords of nuclei in ring
    # using cremer-pople algorithm. It is assumed that the
    # centre of geometry is the centre of the ring.

    Rp = Vec(0, 0, 0)
    Rpp = Vec(0, 0, 0)

    cdef int i
    cdef double ze_angle

    for i in range(n):
        # relative to the ring centroid
        curvec = vec_sub(coords[i], centroid)

        ze_angle = 2.0 * pi * (i - 1) / n
        Rp = vec_scaled_add(Rp, sin(ze_angle), curvec)
        Rpp = vec_scaled_add(Rpp, cos(ze_angle), curvec)

    z = vec_cross(Rp, Rpp)
    z = vec_normalize(z)

    # lmbda = np.dot(z, Rp)
    # y = Rp - z * lmbda
    # y /= np.linalg.norm(y)

    # x = np.cross(y, z)

    # calculate displacement from mean plane

    cdef double displ = 0
    cdef double dot

    for i in range(n):
        # relative to the ring centroid
        curvec = vec_sub(coords[i], centroid)

        dot = vec_dot(curvec, z)
        displ += dot * dot

    return min(sqrt(displ), 2.0)  # truncate amplitude at 2

def ring_calc_pucker_amplitude_py(object ring):
    coords = atoms_coords(ring.atoms)
    return ring_calc_pucker_amplitude(coords.data(), coords.size())

def paperchain_colormap(object ring):
    coords = atoms_coords(ring.atoms)
    cdef double pucker = ring_calc_pucker_amplitude(coords.data(), coords.size())

    cdef np.ndarray ret = np.empty(4, dtype=np.float32)
    p_ret = <float *>ret.data

    # Hot to cold color map:
    # Red -> Yellow -> Green -> Cyan -> Blue -> Magenta
    if pucker < 0.40:
        # Red (1,0,0) -> Yellow (1,1,0)
        p_ret[0] = 1.0  # red
        p_ret[1] = pucker * 2.5  # increase green -> yellow
        p_ret[2] = 0.0
    elif pucker < 0.56:
        # Yellow (1,1,0) -> Green (0,1,0)
        p_ret[0] = 1.0 - (pucker - 0.40) * 6.25  # decrease red -> green
        p_ret[1] = 1.0
        p_ret[2] = 0.0
    elif pucker < 0.64:
        # Green (0,1,0) -> Cyan (0,1,1)
        p_ret[0] = 0.0
        p_ret[1] = 1.0  # green
        p_ret[2] = (pucker - 0.56) * 12.5  # increase blue
    elif pucker < 0.76:
        # Cyan (0,1,1) -> Blue (0,0,1)
        p_ret[0] = 0.0
        p_ret[1] = 1.0 - (pucker - 0.64) * 5.0  # decrease green
        p_ret[2] = 1.0
    else:
        # Blue (0,0,1) -> Magenta (1,0,1)
        p_ret[0] = (pucker - 0.76) * 0.8  # increase red
        p_ret[1] = 0.0
        p_ret[2] = 1.0

    p_ret[3] = 1.0

    return ret

