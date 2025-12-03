from libc.stdint cimport int64_t
from libcpp.vector cimport vector

cdef extern from "<arrays/pythonarray.h>":
    #
    # Convert C arrays to Python Numpy arrays.
    #
    object c_array_to_python(const int *values, int64_t size)
    object c_array_to_python(const vector[int] &values)
    object c_array_to_python(const vector[int64_t] &values)
    object c_array_to_python(const vector[int] &values, int64_t size0, int64_t size1)
    object c_array_to_python(const vector[float] &values)
    object c_array_to_python(const vector[float] &values, int64_t size0, int64_t size1)
    object c_array_to_python(const float *values, int64_t size)
    object c_array_to_python(const double *values, int64_t size)
    object c_array_to_python(const int *values, int64_t size0, int64_t size1)
    object c_array_to_python(const float *values, int64_t size0, int64_t size1)
    object c_array_to_python(const double *values, int64_t size0, int64_t size1)
