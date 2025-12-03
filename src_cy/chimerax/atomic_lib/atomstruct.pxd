from libcpp.vector cimport vector

cdef extern from "<atomstruct/Atom.h>" namespace "atomstruct":
    cdef cppclass Coord:
        double operator[](int)

    cdef cppclass Point:
        Point(double x, double y, double z)

    cdef cppclass Atom:
        Coord &coord() except +
