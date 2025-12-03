# distutils: language=c++
# cython: boundscheck=False, cdivision=True, language_level=3, wraparound=False

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector

from chimerax.atomic_lib.atomstruct cimport Atom

from ._utils cimport *

cdef inline Vec atom_coord(object atom):
    cdef Atom *p = <Atom *><uintptr_t>atom._c_pointer.value

    coord = &p.coord()
    return Vec(deref(coord)[0], deref(coord)[1], deref(coord)[2])

cdef vector[Vec] atoms_coords(object atoms)

cdef Vec ring_get_centroid(const Vec *coords, int n) noexcept

cdef (Vec, Vec) ring_get_centroid_and_normal(const Vec *coords, int n) noexcept

cdef CyFrame ring_get_frame(const Vec *coords, int n) noexcept

cdef double ring_calc_pucker_amplitude(const Vec *coords, int n) noexcept
