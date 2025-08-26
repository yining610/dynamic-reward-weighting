import math
from typing import List

def _to_maximization(p: tuple, maximise: List[bool]) -> tuple:

    return tuple(x if maximise[i] else -x for i, x in enumerate(p))

def _hypervolume(points: List[tuple], ref: tuple) -> float:
    """
    Adapted from Fonseca et al. (2006) recursive slicing algorithm.
    """
    if not points:
        return 0.0

    # 1-D case
    if len(ref) == 1:
        return max(p[0] for p in points) - ref[0]

    points = sorted(points, key=lambda p: p[0])

    hv = 0.0
    ref0 = ref[0] # moving slice position
    while points:
        # Current slice: width along dim-0
        p0 = points[0]
        width = p0[0] - ref0
        if width > 0:
            # All points in this slice, projected to the remaining m-1 dims
            slice_pts = [p[1:] for p in points]
            slice_ref = ref[1:]
            hv += width * _hypervolume(slice_pts, slice_ref)
            ref0 = p0[0]
        # Keep only points strictly beyond the present slice
        points = [p for p in points if p[0] > p0[0]]

    return hv

def compute_score(buffer: List[tuple], current_point: tuple, maximize: List[bool]):

    if len(buffer) == 0:
        raise ValueError("`buffer` must contain at least the reference point.")

    reference_point = buffer[0]
    prev_points     = buffer[1:]

    ref_max   = _to_maximization(reference_point, maximize)
    prev_max  = [_to_maximization(p, maximize) for p in prev_points]
    cur_max   = _to_maximization(current_point,  maximize)

    # Hyper-volume with and without the new point
    hv_without = _hypervolume(prev_max, ref_max)
    hv_with    = _hypervolume(prev_max + [cur_max], ref_max)

    hv_contribution = hv_with - hv_without

    eps = 1e-12
    # NOTE: if the buffer only has one point, it is possible to have the negative HV contribution as the model may get worse from the reference point
    if hv_contribution < -eps and len(buffer) > 1:
        raise ValueError(f"Unexpected negative hypervolumn contribution ({hv_contribution}) for point {current_point} to the buffer {buffer}")
    hv_contribution = max(hv_contribution, 0.0)

    return 0.5 + 1.5 * math.tanh(hv_contribution), hv_contribution