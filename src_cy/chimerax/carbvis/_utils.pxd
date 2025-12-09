# distutils: language=c++
# cython: boundscheck=False, cdivision=True, language_level=3, wraparound=False

import numpy as np

from cython cimport floating
from libc.math cimport sqrt, pow
from libcpp.vector cimport vector

cimport numpy as np

ctypedef fused real:
    short
    int
    long
    float
    double

cdef inline real clip(real x, real a, real b) noexcept:
    """clip x to interval [a, b]"""
    if x < a:
        return a
    elif x > b:
        return b
    else:
        return x

cdef inline bint allclose(
    floating *arr,
    int n,
    floating b,
    floating rtol=1e-05,
    floating atol=1e-08,
) noexcept:
    """np.allclose"""
    cdef floating tol = atol + rtol * abs(b)
    cdef int i
    for i in range(n):
        if abs(arr[i] - b) > tol:
            return False
    return True

cdef packed struct Vec:
    double x
    double y
    double z

cdef inline Vec vec_from_array(const double *arr) noexcept:
    cdef Vec ret
    ret.x = arr[0]
    ret.y = arr[1]
    ret.z = arr[2]
    return ret

cdef inline Vec vec_from_ndarray(np.ndarray arr):
    cdef const double[:] view = arr
    cdef Vec ret
    ret.x = view[0]
    ret.y = view[1]
    ret.z = view[2]
    return ret

cdef inline Vec vec_to_array(const Vec &v, double *arr) noexcept:
    arr[0] = v.x
    arr[1] = v.y
    arr[2] = v.z

cdef inline np.ndarray vec_to_ndarray(const Vec &v):
    cdef np.ndarray ret = np.empty(3, dtype=np.float64)
    vec_to_array(v, <double *>ret.data)
    return ret

cdef inline bint vec_allclose(Vec &v, double b, double rtol=1e-05, double atol=1e-08) noexcept:
    """np.allclose"""
    cdef double tol = atol + rtol * abs(b)
    cdef int i
    if abs(v.x - b) > tol:
        return False
    if abs(v.y - b) > tol:
        return False
    if abs(v.z - b) > tol:
        return False
    return True

cdef inline Vec vec_add(const Vec &a, const Vec &b) noexcept:
    """a + b"""
    cdef Vec ret
    ret.x = a.x + b.x
    ret.y = a.y + b.y
    ret.z = a.z + b.z
    return ret

cdef inline Vec vec_sub(const Vec &a, const Vec &b) noexcept:
    """a - b"""
    cdef Vec ret
    ret.x = a.x - b.x
    ret.y = a.y - b.y
    ret.z = a.z - b.z
    return ret

cdef inline Vec vec_scale(double a, const Vec &b) noexcept:
    """a * b"""
    cdef Vec ret
    ret.x = a * b.x
    ret.y = a * b.y
    ret.z = a * b.z
    return ret

cdef inline Vec vec_scaled_add(const Vec &a, double b, const Vec &c) noexcept:
    """a + b * c"""
    cdef Vec ret
    ret.x = a.x + b * c.x
    ret.y = a.y + b * c.y
    ret.z = a.z + b * c.z
    return ret

cdef inline Vec vec_scaled_sub(const Vec &a, double b, const Vec &c) noexcept:
    """a - b * c"""
    cdef Vec ret
    ret.x = a.x - b * c.x
    ret.y = a.y - b * c.y
    ret.z = a.z - b * c.z
    return ret

cdef inline Vec vec_addscalar(const Vec &a, double b) noexcept:
    """a + b"""
    cdef Vec ret
    ret.x = a.x + b
    ret.y = a.y + b
    ret.z = a.z + b
    return ret

cdef inline Vec vec_negate(const Vec &v) noexcept:
    """-v"""
    cdef Vec ret
    ret.x = -v.x
    ret.y = -v.y
    ret.z = -v.z
    return ret

cdef inline double vec_norm2(const Vec &v) noexcept:
    """get the squared norm of v"""
    return vec_dot(v, v)

cdef inline double vec_norm(const Vec &v) noexcept:
    """get the norm of v"""
    return sqrt(vec_norm2(v))

cdef inline Vec vec_normalize(const Vec &v) noexcept:
    """v / norm(v)"""
    cdef double len2 = vec_norm2(v)
    # prevent division by zero
    if len2 > 0:
        return vec_scale(1 / sqrt(len2), v)
    else:
        return v

cdef inline double vec_dot(const Vec &a, const Vec &b) noexcept:
    """a dot b"""
    return a.x * b.x + a.y * b.y + a.z * b.z

cdef inline Vec vec_cross(const Vec &a, const Vec &b) noexcept:
    """a cross b"""
    cdef Vec ret
    ret.x = a.y * b.z - a.z * b.y
    ret.y = a.z * b.x - a.x * b.z
    ret.z = a.x * b.y - a.y * b.x
    return ret

cdef inline Vec vec_proj(const Vec &a, const Vec &b) noexcept:
    """proj_b a (proj a onto b)"""
    return vec_scale(vec_dot(a, b) / vec_norm2(a), b)

cdef Vec vec_rotate(const Vec &v, const Vec &axis, double angle) noexcept

cdef inline (Vec, Vec) spline(
    const Vec &a,
    const Vec &b,
    const Vec &c,
    const Vec &d,
    const float t,
) noexcept:
    """
    Calculate the position and tangent at point t
    along the spline with co-efficients a, b, c and d.
    spline(t) = ((a * t + b) * t + c) * t + d
    """

    path = vec_scaled_add(d, t, c)
    path = vec_scaled_add(path, t * t, b)
    path = vec_scaled_add(path, t * t * t, a)

    tan = vec_scaled_add(c, 2 * t, b)
    tan = vec_scaled_add(tan, 3 * t * t, a)
    tan = vec_normalize(tan)

    return path, tan

cdef struct CyFrame:
    Vec origin
    Vec up
    Vec forward
    Vec right

cdef void frame_align(CyFrame &frame, const Vec &forward) noexcept

cdef class Frame:
    cdef CyFrame frame

cdef packed struct CyColor:
    float r
    float g
    float b
    float a

cdef inline CyColor color_from_array(const float *arr) noexcept:
    cdef CyColor ret
    ret.r = arr[0]
    ret.g = arr[1]
    ret.b = arr[2]
    ret.a = arr[3]
    return ret

cdef inline CyColor color_to_array(const CyColor &c, float *arr) noexcept:
    arr[0] = c.r
    arr[1] = c.g
    arr[2] = c.b
    arr[3] = c.a

cdef inline CyColor colors_avg(const vector[CyColor] &colors) noexcept:
    """Calculate the average of a list of colors linearly (with gamma=2.2 correction)"""

    cdef int n = colors.size()

    ret = CyColor(0, 0, 0, 0)

    if n == 0:
        return ret

    cdef int i
    for i in range(n):
        color = colors[i]

        # linearize
        ret.r += pow(color.r, 2.2)
        ret.g += pow(color.g, 2.2)
        ret.b += pow(color.b, 2.2)

        # blend alpha directly
        ret.a += color.a

    # average
    ret.r /= n
    ret.g /= n
    ret.b /= n
    ret.a /= n

    # avoid pow() with negative base
    if ret.r < 0:
        ret.r = 0
    if ret.g < 0:
        ret.g = 0
    if ret.b < 0:
        ret.b = 0
    if ret.a < 0:
        ret.a = 0

    # delinearize
    ret.r = pow(ret.r, (1 / 2.2))
    ret.g = pow(ret.g, (1 / 2.2))
    ret.b = pow(ret.b, (1 / 2.2))

    return ret
