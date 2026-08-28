import torch

def lane_point_valid(lanes):
    """Per-point validity of a lane polyline tensor.

    Mirrors the padding convention used in ``LaneFusionEncoder.forward``: a
    point is padding iff all eight geometry channels are exactly zero.

    Args:
        lanes: [B, P, V, D] with D >= 8.
    Returns:
        [B, P, V] bool, True where the point is real.
    """
    return torch.sum(lanes[..., :8] != 0, dim=-1) > 0


def valid_lane_mask(lanes):
    """Lanes holding at least one real point.

    Returns:
        [B, P] bool, True where the lane is valid. This is the complement of the
        ``mask_p`` the encoder returns.
    """
    return lane_point_valid(lanes).any(dim=-1)


def _endpoints(lanes):
    """First and last *valid* point of each lane.

    Padding sits at the tail of a polyline, but taking ``[..., 0, :2]`` and
    ``[..., -1, :2]`` blindly would read padding for short lanes, so the indices
    are recovered from the validity mask.

    Returns:
        (start, end), each [B, P, 2].
    """
    pv = lane_point_valid(lanes)  # [B, P, V]
    V = pv.shape[-1]
    idx = torch.arange(V, device=lanes.device)

    # First valid index; lanes with no valid point fall back to 0 and are
    # excluded by the caller's validity mask anyway.
    first = torch.where(pv, idx, torch.full_like(idx, V)).min(dim=-1).values
    last = torch.where(pv, idx, torch.full_like(idx, -1)).max(dim=-1).values
    first = first.clamp(0, V - 1)
    last = last.clamp(0, V - 1)

    gather = lambda i: torch.gather(
        lanes[..., :2], 2, i[..., None, None].expand(-1, -1, 1, 2)
    ).squeeze(2)
    return gather(first), gather(last)


def contested_lane_mask(lanes, lanes_is_route, valid=None, start_tol_m=2.0,
                        end_min_m=10.0, coord_scale=1.0):
    """Branch lanes whose sibling carries the opposite route label.

    This is the sharpest available proxy for "the decision that decides
    navigation compliance": two lanes leave the same point, one is on route and
    one is not, and the head has to tell them apart.  A branch lane whose
    siblings are all labelled the same way is not actually contested -- the
    whole fork is on route, or none of it is.

    The ground-truth labels select *which* lanes are scored; they are never fed
    to the model, so this is stratification, not leakage.

    Returns:
        [B, P] bool.
    """
    if valid is None:
        valid = valid_lane_mask(lanes)
    if lanes_is_route.dim() == 3:
        lanes_is_route = lanes_is_route.squeeze(-1)

    start, end = _endpoints(lanes)
    ds = torch.cdist(start * coord_scale, start * coord_scale)
    de = torch.cdist(end * coord_scale, end * coord_scale)

    pair_valid = valid[:, :, None] & valid[:, None, :]
    eye = torch.eye(valid.shape[1], dtype=torch.bool, device=lanes.device)
    pair_valid = pair_valid & ~eye[None]

    sibling = (ds < start_tol_m) & (de > end_min_m) & pair_valid
    label = lanes_is_route.bool()
    differs = label[:, :, None] != label[:, None, :]
    return (sibling & differs).any(dim=-1) & valid