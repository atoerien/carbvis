# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language=c++
# cython: boundscheck=False, cdivision=True, language_level=3, wraparound=False

import numpy as np

from libc.math cimport acos, cos, sin

cimport numpy as np

from ._utils cimport *

cdef Vec vec_rotate(const Vec &v, const Vec &axis, double angle) noexcept:
    """Rotate vector v about axis by angle using Rodrigues' formula."""

    norm_axis = vec_normalize(axis)
    cdef double sin_a = sin(angle)
    cdef double cos_a = cos(angle)

    cdef double dot = vec_dot(norm_axis, v)
    cross = vec_cross(norm_axis, v)

    ret = vec_scale(cos_a, v)
    ret = vec_scaled_add(ret, (1 - cos_a) * dot, norm_axis)
    ret = vec_scaled_add(ret, sin_a, cross)
    return ret

cdef void frame_align(CyFrame &frame, const Vec &forward) noexcept:
    """Rotate the frame to align with a forward vector."""

    rot_axis = vec_cross(frame.forward, forward)
    cdef double axis_norm = vec_norm(rot_axis)

    cdef double rot_angle

    # skip rotation if we're already aligned
    if axis_norm > 1e-4:
        rot_angle = acos(
            # float shenanigans
            clip(vec_dot(frame.forward, forward), -1.0, 1.0)
        )

        # rotate frame angle rot_angle about rot_axis
        frame.up = vec_rotate(frame.up, rot_axis, rot_angle)
        frame.forward = vec_rotate(frame.forward, rot_axis, rot_angle)
        frame.right = vec_rotate(frame.right, rot_axis, rot_angle)

cdef class Frame:
    def __init__(
        self,
        np.ndarray origin,
        np.ndarray up,
        np.ndarray forward,
        np.ndarray right,
    ):
        self.frame.origin = vec_from_ndarray(origin)
        self.frame.up = vec_from_ndarray(up)
        self.frame.forward = vec_from_ndarray(forward)
        self.frame.right = vec_from_ndarray(right)

    @property
    def origin(self):
        return vec_to_ndarray(self.frame.origin)

    @origin.setter
    def origin(self, np.ndarray v):
        self.frame.origin = vec_from_ndarray(v)

    @property
    def up(self):
        return vec_to_ndarray(self.frame.up)

    @up.setter
    def up(self, np.ndarray v):
        self.frame.up = vec_from_ndarray(v)

    @property
    def forward(self):
        return vec_to_ndarray(self.frame.forward)

    @forward.setter
    def forward(self, np.ndarray v):
        self.frame.forward = vec_from_ndarray(v)

    @property
    def right(self):
        return vec_to_ndarray(self.frame.right)

    @right.setter
    def right(self, np.ndarray v):
        self.frame.right = vec_from_ndarray(v)

    def copy(self):
        cdef Frame ret = Frame.__new__(Frame)
        ret.frame = self.frame
        return ret

    def align(self, np.ndarray forward):
        """Rotate the frame to align with a forward vector."""
        frame_align(self.frame, vec_from_ndarray(forward))
