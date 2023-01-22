#

# searcher (forced decoding, beam search, ...)

import math
import torch
import torch.nn.functional as F
from zgen.utils import Conf, zwarn, default_pickle_serializer
from zgen.utils import nn as nnutils
from .help_others import *

# --
# BlankStatus:
STATUS_CLOSE = 0
STATUS_OPEN = 1
STATUS_DELAY = 2
# --

# --
# helper to manage cache
# at dim[0] - dim batch
def reindex_cache_db(cache, idx_t):
    for _ln, _lv in cache.items():  # layer
        # todo(note): be careful about this, no need to arrange others (cross) since all the same if beam_size is not changed!
        _cv = _lv['self']
        for _tn, _tv in _cv.items():  # specific tensor
            _cv[_tn] = _tv[idx_t]
    # --

# at dim[-2] - dim length
def reindex_cache_dl(cache, arange_t, idx_t, mask_t):
    for _ln, _lv in cache.items():  # layer
        _cv = _lv['self']
        for _tn, _tv in _cv.items():  # only self part
            if len(_tv.shape) == 4:  # [B, H, L, ...]
                _cv[_tn] = _tv.transpose(1,2)[arange_t, idx_t].transpose(1,2)  # [B, H, L', ...]
            elif len(_tv.shape) == 2:  # [B, L]
                # todo(note): once a bug: forget to further apply mask here!
                _cv[_tn] = _tv[arange_t, idx_t] * mask_t  # [B, L', ...]
            else:
                raise NotImplementedError("UNK cache value!")
    # --

# clear 'self' parts
def clear_cache_self(cache):
    for _ln, _lv in cache.items():  # layer
        if 'self' in _lv:
            _lv['self'].clear()
    # --
# --

# --
# helpers for rposi
# [*, Ls], 2x[*, k], [*, Ls], Li
def get_rposi_incr(stage_t, left_t, right_t, mask_t, imap_t, ilen: int):
    t_v0 = (stage_t >= 0).long()  # [*, Ls]
    t_vL, t_vR = t_v0*0, t_v0*0
    t_vL.scatter_add_(-1, left_t, mask_t.long())  # note: use "scatter_add_" to avoid rewrite idx0!!
    t_vR.scatter_add_(-1, right_t, mask_t.long())
    t_csumL0 = (t_v0 + t_vL).cumsum(-1)  # [*, Ls], position of insertion words
    t_csumL = t_csumL0.gather(-1, left_t)  # [*, k]
    t_csumR0 = (t_v0 + t_vR).cumsum(-1)  # [*, Ls], position of original words
    _tmp_rr0 = nnutils.constants([len(t_csumR0), 1+ilen]).long()  # [*, 1+Li]
    _tmp_rr0.scatter_(-1, imap_t+1, t_csumR0)
    t_csumR = _tmp_rr0[:, 1:]  # [*, Li]
    rr = torch.cat([(t_csumR.unsqueeze(-2) - t_csumL.unsqueeze(-1)),
                    (t_csumL.unsqueeze(-2) - t_csumL.unsqueeze(-1))], -1)  # [*, Ls, Li+Ls]
    return rr
# --
# [*, Lall]
def get_rposi_full(stage_t):
    t_v = (stage_t.unsqueeze(-1) >= stage_t.unsqueeze(-2)) & (stage_t.unsqueeze(-2) >= 0)  # [*, Q, K]
    t_csum = t_v.long().cumsum(-1)  # [*, Q, K]
    rr = t_csum - t_csum.diagonal(dim1=1, dim2=2).unsqueeze(-1)  # [*, Q, K]
    return rr
# --

# record computation caches (in decoding[include only toks] order)
class SearchingCache:
    def __init__(self):
        # mapping
        self.imap_t = None  # [*, ?], canvas-idx to internal-idx
        self.ilen = 0   # internal length
        # posi
        self.use_rel = False
        self.posi_t = None  # [*, ??, D], this needs to be built incrementally!
        # running
        self.mask_t = None  # [*, ??]
        self.hid_t = None  # [*, ??, D]
        self.run_cache = {}  # **[*, ??, D], internal cache for the NN models
        # --

    # clear running parts
    def clear_running_parts(self):
        self.mask_t = None
        # todo(note): [DEBUG] comment this for debug!
        self.hid_t = None
        clear_cache_self(self.run_cache)
        # --

    @classmethod
    def create(cls, bsize, mod, skip_forw: bool, **forw_kwargs):
        ret = cls()
        ret.posi_t, ret.hid_t, ret.mask_t = \
            mod.incr.run_incr_init(bsize, ret.run_cache, skip_forw, **forw_kwargs)  # [*, 2, ?]
        # setup imap_t
        imap_t = nnutils.zeros([bsize, 2], dtype=nnutils.DEFAULT_INT)  # [*, 2]
        imap_t[..., 1] = 1  # [*, [0,1]]
        ret.imap_t = imap_t
        ret.ilen = 2
        ret.use_rel = mod.incr.incr_rel
        # note: no need to compress here!
        return ret

    def append(self, cur_id_t, cur_mask_t, c_left_t, c_right_t, mod, skip_forw: bool, compress_thresh: float, **forw_kwargs):
        # note: need to update imap before this!!
        # translate idxes from canvas ones to inner ones!
        _cur_imap_t = self.imap_t
        inner_left_t, inner_right_t = _cur_imap_t.gather(-1, c_left_t), _cur_imap_t.gather(-1, c_right_t)  # [*, kk]
        # run
        rets = mod.incr.run_incr(cur_id_t, cur_mask_t, inner_left_t, inner_right_t, self.posi_t, self.mask_t, self.hid_t,
                                 self.run_cache, skip_forw, **forw_kwargs)  # [*, A, ?]
        self.posi_t, self.hid_t, self.mask_t = rets
        # compress?
        self.ilen += cur_id_t.shape[-1]
        self.compress(compress_thresh)
        # --

    def compress(self, compress_thresh: float):
        _mask_t = self.mask_t  # [*, S]
        if _mask_t is not None and compress_thresh>0.:
            _new_idxes, _new_mask = nnutils.mask2idx(_mask_t)  # [*, S']
            _orig_size, _new_size = _mask_t.shape[-1], _new_mask.shape[-1]
            if _new_size < (_orig_size * compress_thresh):  # no need to compress if too close
                _arange_t = nnutils.get_arange_t(_mask_t.shape[0], 1)  # [*, 1]
                # [*, S] -> [*, S']
                self.mask_t = _new_mask
                if self.posi_t is not None:
                    self.posi_t = self.posi_t[_arange_t, _new_idxes]
                self.hid_t = self.hid_t[_arange_t, _new_idxes]
                reindex_cache_dl(self.run_cache, _arange_t, _new_idxes, _new_mask)
                # remap imap
                old2new = _mask_t.to(nnutils.DEFAULT_INT).cumsum(-1)-1  # [*, S]
                _imap_mask_b = (self.imap_t<0)
                self.imap_t = old2new[_arange_t, self.imap_t]  # [*, clen]
                self.imap_t[_imap_mask_b] = -1
                assert self.ilen == _orig_size
                self.ilen = _new_size
        # --

    # [*, new]
    def reindex_canvas(self, new2old, new_mask):
        self.imap_t = self.imap_t.gather(-1, new2old)  # [*, new]
        self.imap_t[new_mask <= 0.] = -1
        # --

    # [*]
    def reindex_batch(self, reidx_t):
        self.imap_t = self.imap_t[reidx_t]
        if self.posi_t is not None:
            self.posi_t = self.posi_t[reidx_t]
        if self.mask_t is not None:
            self.mask_t = self.mask_t[reidx_t]
        if self.hid_t is not None:
            self.hid_t = self.hid_t[reidx_t]
        reindex_cache_db(self.run_cache, reidx_t)

# record trajectory (in decoding[include all] order), corresponds to canvas's latest idx
class SearchingHistory:
    def __init__(self):
        self.left_ts = []  # [*, ?], left parent (latest) idx
        self.right_ts = []  # [*, ?], right parent (latest) idx
        self.self_ts = []  # [*, ?] self (latest) idx (or special ones)
        self.mask_ts = []  # [*, ?] mask
        self.score_ts = []  # [*, ?] (optional) predicting logprobs
        # --

    # [*, K]; these should include all, including NOIs
    def append(self, left_t, right_t, self_t, mask_t, score_t):
        self.left_ts.append(left_t)
        self.right_ts.append(right_t)
        self.self_ts.append(self_t)
        self.mask_ts.append(mask_t)
        self.score_ts.append(score_t)

    # [*, old]
    def reindex_canvas(self, old2new):
        for z in [self.left_ts, self.right_ts, self.self_ts]:
            for ii, vv in enumerate(z):
                if vv is not None:
                    z[ii] = old2new.gather(-1, vv)  # [*, ?]
                # note: usually 0 map to 0 thus no need to further mask for safety/pretty-looking
        # --

    # [*]
    def reindex_batch(self, reidx_t):
        for z in [self.left_ts, self.right_ts, self.self_ts]:
            for ii, vv in enumerate(z):
                if vv is not None:
                    z[ii] = vv[reidx_t]
        # --

# record current status, in natural order
class SearchingCanvas:
    def __init__(self):
        self.id_t = None  # [*, L], naturally arranged
        self.mask_t = None  # [*, L]
        self.stage_t = None  # [*, L], starting with 0
        self.status_t = None  # [*, L]: status of the blank (BlankStatus) before
        self.score_t = None  # [*]: accumulated score
        # --
        self.cur_stage = 0  # current stage!
        self.history = SearchingHistory()
        self.cache: SearchingCache = None
        self.svoc = None
        # --

    # finished when all closed!
    def is_finished(self):
        ret = (self.status_t == STATUS_CLOSE).all().item()
        return ret

    # [*, ??]
    def score(self, left_t, right_t, mod, **forw_kwargs):
        cc = self.cache
        # reindex from canvas index to internal index!
        if mod.incr.is_model0:  # note: no need to reindex for full-forward mode!!
            r_left_t, r_right_t = left_t, right_t
            _last_stage_t = None
        else:
            r_left_t, r_right_t = cc.imap_t.gather(-1, left_t), cc.imap_t.gather(-1, right_t)
            # --
            if mod.incr.last_att is not None:
                _last_stage_t = torch.zeros_like(cc.mask_t)  # [*, ??]
                _hit_idx, _hit_mask = nnutils.mask2idx(self.stage_t == self.cur_stage)  # [*, ???], [*, ???]
                _cc_idx = cc.imap_t.gather(-1, _hit_idx) * (_hit_mask.to(nnutils.DEFAULT_INT))  # [*, ???]
                _last_stage_t.scatter_(-1, _cc_idx, 1.)  # [*, ??]
                _last_stage_t[..., 0] = 1.  # note: for simplicity, always attend to idx0=CLS!
            else:
                _last_stage_t = None
            # --
        input_mask_t = self.mask_t * (self.stage_t>=0).to(nnutils.DEFAULT_FLOAT)
        # --
        feats = mod.incr.run_score_feat(cc.hid_t, cc.mask_t, _last_stage_t, self.id_t, input_mask_t, r_left_t, r_right_t,
                                        cache=cc.run_cache, **forw_kwargs)
        scores = mod.incr.run_score_head(feats)
        return scores

    # [*]
    def reindex_batch(self, reidx_t):
        self.id_t = self.id_t[reidx_t]
        self.mask_t = self.mask_t[reidx_t]
        self.stage_t = self.stage_t[reidx_t]
        self.status_t = self.status_t[reidx_t]
        if self.score_t is not None:
            self.score_t = self.score_t[reidx_t]
        if self.history is not None:
            self.history.reindex_batch(reidx_t)
        self.cache.reindex_batch(reidx_t)
        # --

# canvas for forced decoding:
# -> final canvas is fixed and just use that, no need to increase and rearrange!
class SearchingCanvasFd(SearchingCanvas):
    def __init__(self):
        super().__init__()

    @classmethod
    def create(cls, id_t, mask_t, len_t, mod, skip_forw: bool, **forw_kwargs):
        if len_t is None:
            len_t = mask_t.sum(-1, keepdims=True).to(nnutils.DEFAULT_INT)  # [*, 1]
        # state
        stage_t = torch.full_like(id_t, -1)  # [*, L]
        stage_t[..., 0] = 0  # assume stage0 for thes two!
        stage_t.scatter_(-1, len_t-1, 0)
        status_t = torch.full_like(id_t, STATUS_CLOSE)  # [*, L]
        status_t[mask_t>0.] = STATUS_OPEN
        status_t[..., 0] = STATUS_CLOSE  # close before the starting token!
        # init cache
        imap_t = torch.full_like(id_t, -1)  # [*, L]
        imap_t[..., 0] = 0  # idx0=CLS
        len_t_m1 = len_t - 1
        imap_t.scatter_(-1, len_t_m1, 1)  # idx1=SEP
        # --
        ret = cls()
        ret.id_t, ret.mask_t, ret.stage_t, ret.status_t = id_t, mask_t, stage_t, status_t
        ret.cache = SearchingCache.create(id_t.shape[0], mod, skip_forw, **forw_kwargs)  # init cache
        ret.cache.imap_t = imap_t  # replace it with the full one!
        ret.svoc = mod.ztask.special_vocab
        return ret

    # [*, k], canvas idx (should not include NOIs, only valide ones)
    def append(self, left_t, right_t, self_t, mask_t, close_right_t, **cache_kwargs):
        _cur_stage = self.cur_stage + 1
        # --
        # setup status
        self.status_t.scatter_(-1, close_right_t, STATUS_CLOSE)  # note: no need further op, cls(0) is always closed
        # --
        if not (mask_t<=0.).all().item():  # no need to update if no real tokens!
            # rposi
            rposi = get_rposi_incr(
                self.stage_t, left_t, right_t, mask_t, self.cache.imap_t, self.cache.ilen) if self.cache.use_rel else None
            # setup stage
            self_t = (mask_t * self_t).to(nnutils.DEFAULT_INT)  # [*, k]
            self.stage_t.scatter_(-1, self_t, _cur_stage)  # not-real ones are all to idx=0
            self.stage_t[..., 0] = 0  # reset for cls
            # update cache imap
            self.cache.imap_t.scatter_(
                -1, self_t, torch.full_like(self_t, fill_value=self.cache.ilen)+nnutils.get_arange_t(self_t.shape[-1], 0))
            self.cache.imap_t[..., 0] = 0  # reset for cls
            # update cache
            _new_id_t = self.id_t.gather(-1, self_t)  # [*, k]
            _new_id_t[mask_t <= 0.] = self.svoc.pad_token_id
            self.cache.append(_new_id_t, mask_t, left_t, right_t, rposi=rposi, **cache_kwargs)
        # --
        self.cur_stage = _cur_stage
        # --

# canvas for open decoding:
# -> need to incrementally build and rearrange things!
class SearchingCanvasOd(SearchingCanvas):
    def __init__(self):
        super().__init__()

    @classmethod
    def create(cls, bsize, mod, record_history: bool, skip_forw: bool, **forw_kwargs):
        svoc = mod.ztask.special_vocab
        bos_id, eos_id = svoc.cls_token_id, svoc.sep_token_id
        _shape = [bsize, 2]
        # state
        id_t = nnutils.constants(_shape, value=bos_id, dtype=nnutils.DEFAULT_INT)  # [*, 2]
        id_t[..., 1] = eos_id
        mask_t = nnutils.constants(_shape, value=1., dtype=nnutils.DEFAULT_FLOAT)  # [*, 2]
        stage_t = torch.full_like(id_t, 0)  # [*, 2]
        status_t = torch.full_like(id_t, STATUS_OPEN)  # [*, 2]
        status_t[..., 0] = STATUS_CLOSE  # close before the first one!
        # --
        ret = cls()
        ret.svoc = svoc
        ret.id_t, ret.mask_t, ret.stage_t, ret.status_t = id_t, mask_t, stage_t, status_t
        ret.cache = SearchingCache.create(bsize, mod, skip_forw, **forw_kwargs)  # init cache
        if not record_history:
            ret.history = None
        return ret

    # 3x[*, k], [*, k']
    def append(self, left_t, right_t, id_t, mask_t, close_right_t, **cache_kwargs):
        _cur_stage = self.cur_stage + 1
        # --
        # setup status
        self.status_t.scatter_(-1, close_right_t, STATUS_CLOSE)  # note: no need further op, cls(0) is always closed
        # --
        invalid_tb = (mask_t<=0.)
        if not invalid_tb.all().item():  # no need to update if no real tokens!
            # rposi
            rposi = get_rposi_incr(
                self.stage_t, left_t, right_t, mask_t, self.cache.imap_t, self.cache.ilen) if self.cache.use_rel else None
            # --
            # update cache imap
            _tmp_idxes = torch.full_like(id_t, fill_value=self.cache.ilen) + nnutils.get_arange_t(id_t.shape[-1], 0)  # [*, k]
            _tmp_idxes[invalid_tb] = -1  # pad
            self.cache.imap_t = torch.cat([self.cache.imap_t, _tmp_idxes], -1)  # [*, prev+k]
            # update cache
            self.cache.append(id_t, mask_t, left_t, right_t, rposi=rposi, **cache_kwargs)
            # --
            # setup canvas (reindex to natural order!)
            # concat to temp
            _mask_ti = mask_t.to(nnutils.DEFAULT_INT)
            tmp_id_t = torch.cat([self.id_t, id_t], -1)  # [*, prev+k]
            # tmp_mask_t = torch.cat([self.mask_t, mask_t], -1)  # [*, prev+k]
            # [*, prev+k]; mask*stage + (1.-mask)*(-1) = mask*(1+stage)-1
            tmp_stage_t = torch.cat([self.stage_t, _mask_ti*(1+_cur_stage)-1], -1)
            # [*, prev+k]; mask*OPEN + (1.-mask)*CLOSE = mask*1
            tmp_status_t = torch.cat([self.status_t, _mask_ti], -1)
            # whether insert before at each idx?
            insert_before = torch.zeros_like(self.id_t)  # [*, prev]
            insert_before.scatter_(-1, right_t, 1)
            insert_before[..., 0] = 0  # [*, prev], no insert at the very beginning!
            old_clen = self.id_t.shape[-1]
            # --
            mask1, mask0 = self.mask_t, (insert_before > 0).to(nnutils.DEFAULT_FLOAT)  # [*, prev]
            idx1 = nnutils.get_arange_t(old_clen, 0).unsqueeze(0).expand_as(self.id_t)  # [*, prev]
            idx0 = (insert_before.cumsum(-1) + (old_clen-1)) * insert_before  # [*, prev]
            _full_shape = list(idx0.shape)[:-1] + [2*old_clen]
            _full_mask = torch.stack([mask0, mask1], -1).view(_full_shape)  # [*, 2*prev]
            _full_idx = torch.stack([idx0, idx1], -1).view(_full_shape)  # [*, 2*prev]
            _full_sel_idx, new_mask = nnutils.mask2idx(_full_mask)  # [*, new]
            new2old = _full_idx.gather(-1, _full_sel_idx)  # [*, new], for each new-idx, what is old-idx? pad by 0
            # update to new canvas idx
            invalid_b = (new_mask <= 0.)
            self.id_t = tmp_id_t.gather(-1, new2old)  # [*, new]
            self.id_t[invalid_b] = self.svoc.pad_token_id
            self.mask_t = new_mask  # [*, new]
            self.stage_t = tmp_stage_t.gather(-1, new2old)  # [*, new]
            self.stage_t[invalid_b] = -1
            self.status_t = tmp_status_t.gather(-1, new2old)  # [*, new]
            self.status_t[invalid_b] = STATUS_CLOSE
            # update history
            if self.history is not None:
                _full_mask_csum = _full_mask.to(nnutils.DEFAULT_INT).cumsum(-1) - 1  # [*, 2*prev]
                _new_idx0, _new_idx1 = _full_mask_csum[..., ::2], _full_mask_csum[..., 1::2]  # [*, prev]
                _new_sel_idx, _ = nnutils.mask2idx(insert_before>0)  # [*, k]
                _new_idx0 = _new_idx0.gather(-1, _new_sel_idx)  # [*, k]
                old2new = torch.cat([_new_idx1, _new_idx0], -1)  # [*, prev+k]
                self.history.reindex_canvas(old2new)
            # update cache's imap
            if self.cache is not None:
                self.cache.reindex_canvas(new2old, new_mask)
        # --
        self.cur_stage = _cur_stage

# --
class Searcher:
    def __init__(self, mod):
        self.mod = mod
        # --
        # read some other things

    # [*, L], special random sampling for model0
    def step_random_decoding(self, ids_t, masks_t, curriculum: float):
        mod = self.mod
        svoc = mod.ztask.special_vocab
        # --
        _shape = list(ids_t.shape)
        _low = max(1.-curriculum, 0.)
        cur_mrates = _low + (1.-_low) * nnutils.rand(_shape[:-1]).unsqueeze(-1)  # [*, 1], mask rate
        # --
        # simple random rollin
        kept_mask_b = ((masks_t>0) & ((nnutils.rand(_shape) >= cur_mrates) | (ids_t == svoc.cls_token_id)
                                      | (ids_t == svoc.sep_token_id)))  # [*, L]
        # --
        len_t = masks_t.sum(-1, keepdims=True).to(nnutils.DEFAULT_INT)  # [*, 1]
        canvas = SearchingCanvasFd.create(ids_t, masks_t, len_t, mod, skip_forw=True)
        canvas.stage_t.fill_(1)
        canvas.stage_t[kept_mask_b] = 0  # make it only two stage
        canvas.status_t.fill_(STATUS_CLOSE)
        # obtain blanks: [*, S]
        one_left_t, one_right_t, one_vmask_t, _, _ = valid2seg(
            (canvas.stage_t==0), len_t, exclude_ending=True, ret_seg=False)
        # note: simply put 0 for self_t
        canvas.history.append(one_left_t, one_right_t, torch.zeros_like(one_left_t), one_vmask_t, None)
        return canvas

    # [*, L]
    def forced_decoding(self, ids_t, masks_t, rollin_specs, rollin_scores, rollin_noise: float, rollin_tau: float,
                        compress_thresh: float, close_only_all_noi: bool, smk_low: float, debug_do_forw: bool, **forw_kwargs):
        mod = self.mod
        conf = mod.conf
        svoc = mod.ztask.special_vocab
        _input_len = max(1, ids_t.shape[-1])  # input length
        # --
        # some specs
        rstra_model, rstra_bt, rstra_ps, rstra_alpha = rollin_specs
        skip_forw = (not rstra_model) and (not debug_do_forw)
        # prepare
        len_t = masks_t.sum(-1, keepdims=True).to(nnutils.DEFAULT_INT)  # [*, 1]
        canvas = SearchingCanvasFd.create(ids_t, masks_t, len_t, mod, skip_forw=skip_forw, **forw_kwargs)
        _arange_t = nnutils.get_arange_t(ids_t.shape[-1], 0)  # [L]
        # --
        # start the rolling
        while True:
            # obtain blanks: [*, k]; (pad: left=0, right=0)
            one_left_t, one_right_t, one_vmask_t, _, _ = valid2seg(
                (canvas.stage_t>=0), len_t, exclude_ending=True, right_close_t=(canvas.status_t==STATUS_CLOSE), ret_seg=False)
            # sample out certain ones
            if smk_low < 1.:
                one_vmask_t = self._sample_masks(one_vmask_t, smk_low, 1.)  # [*, k]
            # select mask: [*, k, L]
            sel_cand_masks = ((one_left_t.unsqueeze(-1) < _arange_t) & (_arange_t < one_right_t.unsqueeze(-1)))\
                .to(nnutils.DEFAULT_FLOAT)  # [*, k, L]; valid candidates
            # obtain scores for selecting: [*, k, L]
            if rstra_model:  # get model scores: [*, k, *]
                _sel_score_tV = canvas.score(one_left_t, one_right_t, mod, **forw_kwargs)
                _sel_score_t = select_scores(_sel_score_tV, ids_t)  # [*, k, L]
            elif rstra_bt:  # balance tree (middle first)
                _center = ((_arange_t * sel_cand_masks).sum(-1) / sel_cand_masks.sum(-1).clamp(min=1)).unsqueeze(-1)  # [*, k, 1]
                _sel_score_t = - (_center - _arange_t).abs()  # [*, k, L], distance to center
            elif rstra_ps:  # pair-scoring
                # _sel_score_t = rollin_scores.unsqueeze(-3)  # [*, 1, L, L]
                _sel_score_t = rollin_scores.unsqueeze(1)  # [*, 1, ...], note: it may be 2d or 1d behind
            else:
                _sel_score_t = rollin_scores.unsqueeze(-2)  # [*, 1, L]
            _sel_score_t = _sel_score_t * rstra_alpha  # rev or not; note: now all rev it here!
            # post processing for rollin-oracle
            _sel_score_t = mod.ps_helper.score(_sel_score_t, sel_cand_masks, one_left_t, one_right_t, canvas.stage_t)  # [*, k, L]
            # special alpha
            if conf.span_size_alpha>0.:
                _sel_score_t = _sel_score_t / (sel_cand_masks.sum(-1, keepdims=True).clamp(min=1) ** conf.span_size_alpha)
            # rollin noise
            if rollin_noise > 0.:
                _sel_score_t = _sel_score_t + nnutils.rand(_sel_score_t.shape) * rollin_noise  # [*, k, L]
            # todo(+N): argmax vs sample? selection of which blank??
            if rollin_tau == 0.:  # argmax
                # add -inf for invalid ones; (pad: -inf or ?? when all row no good)
                _sel_score_t = _sel_score_t + get_pad_values(sel_cand_masks)  # [*, k, L]
                _, sel_idxes_t = _sel_score_t.max(-1)  # [*, k]
            else:
                _sel_score_t = _sel_score_t.log_softmax(-1) / rollin_tau
                _sel_score_t = _sel_score_t + get_pad_values(sel_cand_masks)  # [*, k, L]
                _, sel_idxes_t = nnutils.category_sample(_sel_score_t, keepdim=False)  # [*, K, 1]
            # select!; (pad: 0)
            _sel_noclose_t = (one_right_t > (one_left_t+1)).to(nnutils.DEFAULT_FLOAT)
            _sel_mask_t = one_vmask_t * _sel_noclose_t  # [*, k]
            sel_idxes_t[_sel_mask_t<=0.] = 0  # make sure no strange idxes!
            # prepare real idxes
            _new_mask, _vv = select_and_compress(_sel_mask_t, [sel_idxes_t, one_left_t, one_right_t], pad=0)  # [*, kk]
            _new_idxes_t, _new_left_t, _new_right_t = _vv
            # update history
            canvas.history.append(one_left_t, one_right_t, sel_idxes_t, one_vmask_t, None)
            # update canvas (& cache)
            if close_only_all_noi:  # only close all at once when all NOI!
                close_right_t = one_right_t * (one_right_t <= (one_left_t+1)).all(-1, keepdims=True).to(nnutils.DEFAULT_INT)
            else:
                close_right_t = one_right_t * (one_vmask_t * (1.-_sel_noclose_t)).to(nnutils.DEFAULT_INT)
            canvas.append(_new_left_t, _new_right_t, _new_idxes_t, _new_mask, close_right_t,
                          mod=mod, skip_forw=skip_forw, compress_thresh=compress_thresh, **forw_kwargs)
            # finished?
            if canvas.is_finished():
                break
            if canvas.cur_stage >= (5 * _input_len + 1):  # check for looping
                zwarn(f"Seems to be looping here, break this: {canvas.cur_stage} {ids_t.shape}")
                # default_pickle_serializer.to_file(canvas, "_debug.pkl")  # store states to a local file
                torch.save(canvas, '_debug.pt')
                break
        # --
        if debug_do_forw:
            canvas.cache._tmp_hid_t = canvas.cache.hid_t  # store it to check later!
        canvas.cache.clear_running_parts()  # cleaning!
        return canvas  # p torch.stack([rollin_scores.long(), canvas.stage_t], -1)[8]

    # [*], note: forw_kwargs should extend to beam_size at outside!!
    def open_decoding(self, bsize, beam_size: int, max_len_t, noi_penalty: float, mid_div: str, final_div: str,
                      record_history: bool, compress_thresh: float, close_only_all_noi: bool, dtp: float, **forw_kwargs):
        mod = self.mod
        conf = mod.conf
        svoc = mod.ztask.special_vocab
        pad_id, noi_id = svoc.pad_token_id, svoc.noi_token_id
        _NEG_INF = -10000.
        _PRIZE = 100.
        # --
        # div strategy for middle and final
        mid_div_nope, mid_div_tok, mid_div_ins = [mid_div==z for z in ["nope", "tok", "ins"]]
        final_div_nope, final_div_tok, final_div_ins = [final_div==z for z in ["nope", "tok", "ins"]]
        # --
        # extend canvas at the beginning: [bsize * beam_size, ...]
        canvas = SearchingCanvasOd.create(bsize*beam_size, mod, record_history=record_history, skip_forw=False, **forw_kwargs)
        # prepare special masks: [0, -inf, ...]
        _beam_mask_v = nnutils.constants([beam_size], value=_NEG_INF)  # [B]
        _beam_mask_v[0] = 0.
        _beam_mask_v2 = nnutils.constants([beam_size+1], value=_NEG_INF)  # [B]
        _beam_mask_v2[0] = 0.
        _arange_b2_t = nnutils.get_arange_t(bsize*beam_size, 1)  # [*, 1]
        _arange_bs_t = nnutils.get_arange_t(bsize, 1)  # [bs, 1]
        # --
        # some states
        _shape_b2, _shape_b3 = [bsize, beam_size], [bsize, beam_size, beam_size]
        _prize_t = nnutils.input_real([_PRIZE] + [0.]*(beam_size-1))  # [B]
        accu_score_t = nnutils.zeros(_shape_b2, dtype=torch.float32)  # not: use float32 to record this!!
        # --
        # start looping
        # while True:
        for _ss in range(conf.test_max_step):
            # --
            # obtain the non-closed blanks: [*, k]
            _cur_valids = (canvas.status_t != STATUS_CLOSE).to(nnutils.DEFAULT_FLOAT)  # [*, clen]
            one_right_t, one_vmask_t = nnutils.mask2idx(_cur_valids)  # [*, k]
            one_left_t = (one_right_t - 1).clamp(min=0)  # [*, k]
            # score the blanks: [*, k, V]
            _score_tV = canvas.score(one_left_t, one_right_t, mod, **forw_kwargs)
            _logprob_tV = _score_tV.log_softmax(-1) * (one_vmask_t.unsqueeze(-1))  # [*, k, V]
            _logprob_tV[..., pad_id] = _NEG_INF  # can never select this one!!
            if noi_penalty != 0.:  # directly change this on logprobs!
                _logprob_tV[..., noi_id] += noi_penalty
            # =====
            if conf.test_do_sample:  # simpler sampling
                assert dtp < 0.
                inner_id, inner_score = self._inner_sample(_logprob_tV, one_vmask_t, pad_id, conf)  # [*, k, 1]
                re_id, re_score = inner_id.squeeze(-1), inner_score.squeeze(-1).sum(-1)  # [*, k], [*]
                re_score0 = re_score.view(_shape_b2)  # [bs, B]
                re_left_t, re_right_t, re_vmask_t = one_left_t, one_right_t, one_vmask_t  # [*, k]
            else:  # beam selection!
                # step 0: inner beam search:
                if dtp < 0.:  # [*, B, k], [*, B]
                    inner_id, inner_score = self._inner_beam(_logprob_tV, one_vmask_t, beam_size, pad_id, _beam_mask_v, _arange_b2_t)
                else:  # [*, B, k], [*, B], [*, B, k]
                    inner_id, inner_score = \
                        self._inner_beam_delay(_logprob_tV, one_vmask_t, beam_size, pad_id, _beam_mask_v2, _arange_b2_t, dtp)
                # step 1: outer beam search
                # prepare div-scores
                if mid_div_nope:
                    _ranking_div = nnutils.constants(_shape_b3, 1.)  # [bs, B0, B1]
                elif mid_div_tok:
                    _ranking_div0 = canvas.mask_t.sum(-1).view(_shape_b2).unsqueeze(-1) - 2  # [bs, B0, 1]
                    _ranking_div1 = (((inner_id != noi_id) & (inner_id != pad_id)).to(
                        nnutils.DEFAULT_FLOAT) * one_vmask_t.unsqueeze(-2)).sum(-1)  # [bs*B0, B1]
                    _ranking_div = _ranking_div0 + _ranking_div1.view(_shape_b3)  # [bs, B0, B1]
                elif mid_div_ins:
                    _ranking_div1 = ((inner_id != pad_id).to(nnutils.DEFAULT_FLOAT) * one_vmask_t.unsqueeze(-2)).sum(-1)  # [bs*B0, B1]
                    _ranking_div = (canvas.mask_t.sum(-1) + (canvas.mask_t * (canvas.status_t == STATUS_CLOSE).to(nnutils.DEFAULT_FLOAT)).sum(-1)).view(_shape_b2).unsqueeze(-1) + _ranking_div1.view(_shape_b3) - 3  # [bs, B0, 1], all insertions (including NOIs)
                else:
                    raise NotImplementedError(f"UNK div method {mid_div}")
                _ranking_div.clamp_(min=1.)  # avoid div 0
                # outer beam
                _rr_b0, _rr_b1 = self._outer_beam(accu_score_t, inner_score, _ranking_div, canvas,
                                                  bsize, beam_size, _shape_b3, _prize_t)  # [bs, B0], [bs, B1]
                # --
                # reindex all at the batch dimension!!
                if beam_size != 1:
                    batch_reidx_t = (_rr_b0 + _arange_bs_t * beam_size).view([-1])  # [*]
                    canvas.reindex_batch(batch_reidx_t)
                    accu_score_t = accu_score_t[_arange_bs_t, _rr_b0]  # [bs, B]
                    re_left_t, re_right_t, re_vmask_t = [z[batch_reidx_t] for z in [one_left_t, one_right_t, one_vmask_t]]  # [*, k]
                else:  # no need to re-arrange!
                    re_left_t, re_right_t, re_vmask_t = one_left_t, one_right_t, one_vmask_t
                # gather the corresponding ones
                re_id0, re_score0 = inner_id.view(_shape_b3 + [-1])[_arange_bs_t, _rr_b0, _rr_b1], \
                                    inner_score.view(_shape_b3)[_arange_bs_t, _rr_b0, _rr_b1]  # [bs, B, k], [bs, B]
                re_id, re_score = re_id0.view_as(re_vmask_t), re_score0.view([-1])  # [*, k], [*]
            # =====
            # prepare new ids and scores
            re_vmask_t *= (re_id != pad_id).to(nnutils.DEFAULT_FLOAT)  # exclude PAD for delayed ones!
            accu_score_t += re_score0  # [bs, B]
            # --
            # update history
            if record_history:  # todo(+N): calculate self_t with _new_* tmp idxes?
                canvas.history.append(re_left_t, re_right_t, None, re_vmask_t, re_score)
            # prepare real ones, excluding NOIs
            cur_close_t = (re_id == noi_id).to(nnutils.DEFAULT_FLOAT)  # [*, k]
            cur_tok_t = (1. - cur_close_t) * re_vmask_t  # [*, k]
            _new_mask, _vv = select_and_compress(cur_tok_t, [re_id, re_left_t, re_right_t], pad=[pad_id, 0, 0])  # [*, kk]
            _new_id_t, _new_left_t, _new_right_t = _vv  # [*, kk]
            # update canvas
            if close_only_all_noi:  # close all at once only when all NOI!
                _close_right_t = re_right_t * (cur_tok_t<=0.).all(-1, keepdims=True).to(nnutils.DEFAULT_INT)
            else:
                _close_right_t = re_right_t * cur_close_t.to(nnutils.DEFAULT_INT)
            canvas.append(_new_left_t, _new_right_t, _new_id_t, _new_mask, _close_right_t,
                          mod=mod, skip_forw=False, compress_thresh=compress_thresh, **forw_kwargs)
            # check length limit
            over_limit_tb = (canvas.mask_t.sum(-1) >= max_len_t)  # [*]
            canvas.status_t[over_limit_tb] = STATUS_CLOSE  # close those exceeding limit (force it without NOI)!
            # finished?
            if canvas.is_finished():
                break
        # --
        # final sort
        canvas.score_t = accu_score_t.view([-1])  # [*]
        # rerank according to div
        if beam_size != 1:
            if final_div_nope:
                _ranking_div = nnutils.constants(_shape_b2, 1.)  # [bs, B]
            elif final_div_tok:
                _ranking_div = canvas.mask_t.sum(-1).view(_shape_b2) - 2  # [bs, B]
            elif final_div_ins:
                _ranking_div = (((canvas.status_t == STATUS_CLOSE).to(nnutils.DEFAULT_FLOAT) * canvas.mask_t).sum(-1)
                                + canvas.mask_t.sum(-1)).view(_shape_b2) - 3  # [bs, B], all insertions (including NOIs)
            else:
                raise NotImplementedError(f"UNK div method {final_div}")
            _ranking_div.clamp_(min=1.)  # avoid div 0
            _ranking_score = (accu_score_t / _ranking_div)  # [bs, B]
            _, _rr_idx = _ranking_score.topk(beam_size, dim=-1, sorted=True)  # [bs, B]
            final_batch_reidx_t = (_rr_idx + _arange_bs_t * beam_size).view([-1])  # [*]
            canvas.reindex_batch(final_batch_reidx_t)
        # --
        canvas.cache.clear_running_parts()  # cleaning!
        return canvas

    # --
    # helpers for beam search:

    # [*, k, B]
    def _inner_beam_loop(self, beam_size: int, _shape, _top_scores, _top_ids, _arange_b2_t):
        _shape0, _shape1 = _shape[:-1], _shape[-1]  # [*], k
        _s0_id, _s0_score = None, None  # [*, B, k], [*, B]; (sorted)
        for kk in range(_shape1):
            _cur_top_id, _cur_top_score = _top_ids[..., kk, :], _top_scores[..., kk, :]  # [*, B]
            if kk == 0:
                _s0_id, _s0_score = _cur_top_id.unsqueeze(-1), _cur_top_score  # [*, B, 1], [*, B]
            else:
                _expand_score = _s0_score.unsqueeze(-1) + _cur_top_score.unsqueeze(-2)  # [*, B0, B1]
                _expand_score = _expand_score.view(_shape0 + [beam_size * beam_size])  # [*, B0xB1]
                _t2_score, _t2_idx = _expand_score.topk(beam_size, dim=-1, sorted=True)  # [*, B]
                _sel_b0, _sel_b1 = _t2_idx // beam_size, _t2_idx % beam_size  # [*, B]
                # update: [*, B, ?+1]
                _s0_id = torch.cat([_s0_id[_arange_b2_t, _sel_b0], _cur_top_id[_arange_b2_t, _sel_b1].unsqueeze(-1)], -1)
                _s0_score = _s0_score[_arange_b2_t, _sel_b0] + _cur_top_score[_arange_b2_t, _sel_b1]
        # --
        return _s0_id, _s0_score  # [*, B, k], [*, B]

    # [*, k, V], [*, k]
    def _inner_sample(self, scores_t, mask_t, _pad_id, _conf):
        # inner beam (each sample 1)
        _, inner_id = nnutils.category_sample(
            scores_t, dim=-1, keepdim=True, top_k=_conf.test_sample_topk, top_p=_conf.test_sample_topp)  # [*, k, 1]
        invalid_t = (mask_t <= 0.)
        inner_id[invalid_t] = _pad_id  # [*, k, 1], put pad
        inner_score = scores_t.gather(-1, inner_id)  # [*, k, 1]
        inner_score[invalid_t] = 0.  # [*, k, 1]
        return inner_id, inner_score

    # [*, k, V], [*, k],
    def _inner_beam(self, scores_t, mask_t, beam_size: int, _pad_id, _beam_mask_v, _arange_b2_t):
        _NEG_INF = -10000.
        _shape = list(mask_t.shape)  # [*, k]
        # get top scores (and considering mask)
        _top_scores, _top_ids = scores_t.topk(beam_size, dim=-1, sorted=True)  # [*, k, B]
        _mask_tb = (mask_t <= 0.)
        _top_ids[_mask_tb] = _pad_id  # [*, k, B], put PAD
        _top_scores[_mask_tb] = _beam_mask_v  # [*, k, B], put special ones
        # beam loop
        _s0_id, _s0_score = self._inner_beam_loop(beam_size, _shape, _top_scores, _top_ids, _arange_b2_t)
        return _s0_id, _s0_score  # [*, B, k], [*, B]

    # inner-beam with SMK(search-minimal-keep)
    # [*, k, V], [*, k],
    def _inner_beam_delay(self, scores_t, mask_t, beam_size: int, _pad_id, _beam_mask_v, _arange_b2_t, _dtp: float):
        _NEG_INF = -10000.
        _delay_score = math.log(_dtp) if _dtp>0 else _NEG_INF  # -inf if <=0.
        _nop_id = -1  # special one!
        # --
        _shape = list(mask_t.shape)  # [*, k]
        _shape0, _shape1 = _shape[:-1], _shape[-1]  # [*], k
        # get top scores (and considering mask)
        _top_scores, _top_ids = scores_t.topk(beam_size, dim=-1, sorted=True)  # [*, k, B]
        _top_scores = F.pad(_top_scores, [1,0], value=_delay_score)  # [*, k, 1+B]
        _top_ids = F.pad(_top_ids, [1,0], value=_nop_id)  # [*, k, 1+B]
        # --
        _mask_tb = (mask_t <= 0.)
        _top_ids[_mask_tb] = _pad_id  # [*, k, B], put PAD
        _top_scores[_mask_tb] = _beam_mask_v  # [*, k, B], put special ones
        # --
        # beam loop: [*, B+1, k], [*, B+1]
        _s0_id, _s0_score = self._inner_beam_loop(beam_size+1, _shape, _top_scores, _top_ids, _arange_b2_t)
        _valid_count = mask_t.long().sum(-1).unsqueeze(-1)  # [*, 1]
        invalid_lines = (((_s0_id == _nop_id).sum(-1) >= _valid_count) & (_valid_count > 0))  # [*, B+1]
        _s0_score[invalid_lines] = _NEG_INF
        # retake topk!
        _, _rs_idx = _s0_score.topk(beam_size, dim=-1, sorted=True)  # [*, B]
        ret_id, ret_score = _s0_id[_arange_b2_t, _rs_idx], _s0_score[_arange_b2_t, _rs_idx]  # [*, B, k], [*, B]
        # final modification
        _is_nop = (ret_id == _nop_id)  # [*, B, k]
        ret_id[_is_nop] = _pad_id  # change to pad
        ret_score -= _delay_score * _is_nop.to(nnutils.DEFAULT_FLOAT).sum(-1)
        # --
        return ret_id, ret_score

    # [bs, B0], [bs*B0, B1], [bs, B0, B1] -> [*, B0], [*, B1]
    def _outer_beam(self, accu_score_t, inner_score, ranking_div, canvas,
                    bsize: int, beam_size: int, _shape_b3, _prize_t):
        _NEG_INF = -10000.
        # prepare extra score for ranking
        _extra_ranking_score = nnutils.zeros(_shape_b3)  # [bs, B0, B1] temp scored added for special treatment!
        if canvas.cur_stage == 0:  # mask out non0s at the first step!
            _extra_ranking_score[:, 1:] = _NEG_INF
        one_is_finished_b = (canvas.status_t == STATUS_CLOSE).all(-1).view(_shape_b3[:2])  # [bs, B0]
        _extra_ranking_score[one_is_finished_b] += _prize_t  # keep finished ones as they are!
        # rank them
        _ranking_score0 = accu_score_t.unsqueeze(-1) + inner_score.view(_shape_b3) + _extra_ranking_score  # [bs, B0, B1]
        _ranking_score = (_ranking_score0 / ranking_div).view([bsize, -1])  # [bs, B0xB1]
        _, _rr_idx = _ranking_score.topk(beam_size, dim=-1, sorted=True)  # [bs, B]
        _rr_b0, _rr_b1 = _rr_idx // beam_size, _rr_idx % beam_size  # [bs, B0], [bs, B1]
        return _rr_b0, _rr_b1  # [bs, B0], [bs, B1]

    # sample certain number of masks
    def _sample_masks(self, vmask_t, keep_low: float, keep_high: float):
        # prepare sel number
        _shape = list(vmask_t.shape)  # [*, K]
        _keep_t = keep_low + (keep_high - keep_low) * nnutils.rand(_shape[:-1]).unsqueeze(-1)  # [*, 1]
        _topk_t = 1 + (vmask_t.sum(-1, keepdims=True) * _keep_t).to(nnutils.DEFAULT_INT)  # [*, 1]
        _topk_t.clamp_(min=1, max=_shape[-1])  # make it \in [1,k]
        # select them
        ret = nnutils.select_topk(nnutils.rand(_shape), _topk_t, mask_t=vmask_t)  # [*, K]
        return ret

# --
# b zgen/model/tasks/dec_ilm/help_search:??
