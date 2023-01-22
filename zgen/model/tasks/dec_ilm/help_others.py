#

# some helping routines

__all__ = [
    "append_slice", "select_and_compress", "valid2seg", "get_pad_values", "select_scores",
]

import torch
import torch.nn.functional as F
from zgen.utils import nn as nnutils

# --
# append one slice of idxes; eg: [0, 1, 3, 4, x, x] -> [0, 1, 3, 4, L, x, x]
def append_slice(input_t, masks_t, app_t, pad=0):
    # expand another idx as the seq length (as sentinel)
    ret_t = F.pad(input_t, [0, 1], value=pad)  # [*, k+1]
    ret_t.scatter_(-1, masks_t.sum(-1, keepdims=True).to(nnutils.DEFAULT_INT), app_t)  # [*, k+1]
    return ret_t

# --
# select and compress
def select_and_compress(masks_t, values, pad=None):
    _tmp_idxes, ret_masks_t = nnutils.mask2idx(masks_t)  # [*, k']
    ret_values = [z.gather(-1, _tmp_idxes) for z in values]
    if pad is not None:
        _mm = (ret_masks_t <= 0.)
        if isinstance(pad, int):
            pad = [pad] * len(ret_values)
        for vv, pp in zip(ret_values, pad):
            vv[_mm] = pp
    return ret_masks_t, ret_values

# --
# valid2seg(valid to segments); eg: [1, 0, 1, 1, 0, 1] -> [[0, 2, 3, 5], [2, 3, 5, 6]], ...
def valid2seg(valid_t, len_t, exclude_ending=True, left_close_t=None, right_close_t=None, ret_seg=False):
    # obtain the original segments
    left_t0, vmask_t0 = nnutils.mask2idx(valid_t)  # [*, k0], current left boundaries
    _extend_t0 = append_slice(left_t0, vmask_t0, len_t)  # [*, k0+1]
    right_t0 = _extend_t0[..., 1:]  # [*, k0], right boundaries
    # further filter
    if exclude_ending or (left_close_t is not None) or (right_close_t is not None):
        if exclude_ending:  # no ending piece
            vmask_t0 *= (right_t0 < len_t).to(nnutils.DEFAULT_FLOAT)
        if left_close_t is not None:  # filter left
            vmask_t0 *= (1. - left_close_t.to(nnutils.DEFAULT_FLOAT).gather(-1, left_t0))
        if right_close_t is not None:  # filter right
            right_close_t2 = F.pad(right_close_t.to(nnutils.DEFAULT_FLOAT), [0, 1], value=1.)  # [*, ?+1]
            vmask_t0 *= (1. - right_close_t2.gather(-1, right_t0))
        # rearrange again!
        vmask_t, _vv_t = select_and_compress(vmask_t0, [left_t0, right_t0], pad=0)  # [*, k]
        left_t, right_t = _vv_t
    else:
        vmask_t, left_t, right_t = vmask_t0, left_t0, right_t0
    # get explicit segments?
    if ret_seg:
        if nnutils.is_zero_shape(left_t):  # []
            _max_seg_size = 1
        else:
            _max_seg_size = max(1, int((right_t - left_t).max().item()))
        seg_idxes_t = nnutils.arange(_max_seg_size) + left_t.unsqueeze(-1)  # [*, k, S]
        seg_mask_t = (seg_idxes_t < right_t.unsqueeze(-1)).to(nnutils.DEFAULT_FLOAT)  # [*, k, S], valid ones
        seg_idxes_t[seg_mask_t <= 0.] = 0  # make it safe with 0
    else:
        seg_idxes_t, seg_mask_t = None, None
    # --
    return left_t, right_t, vmask_t, seg_idxes_t, seg_mask_t

# add pad values
def get_pad_values(mask_t, fill_v=float('-inf'), fill_row_v=0.):
    _extra_ones = torch.zeros_like(mask_t)
    _extra_ones[mask_t <= 0.] = fill_v
    if fill_row_v is not None:
        _extra_ones[mask_t.sum(-1) == 0.] = 0.  # avoid all -inf rows
    return _extra_ones

# gather with unsqueeze
def select_scores(scores_t, ids_t):
    _shape0 = list(scores_t.shape)
    _shape1 = list(ids_t.shape)
    for _ in range(len(_shape0) - len(_shape1)):
        ids_t = ids_t.unsqueeze(-2)  # [*, *1, L]
    _sel_t = ids_t.expand(_shape0[:-1] + [_shape1[-1]])  # [*, ??, L]
    ret = scores_t.gather(-1, _sel_t)  # [*, ??, L]
    return ret

# --
