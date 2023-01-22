#

# helper for incremental building

__all__ = [
    "IncrCacheFd", "IncrEmbedConf", "IncrEmbed",
]

import torch
from zgen.utils import ConfEntryChoices
from zgen.utils import nn as nnutils
from ...core import *

# For forced decoding!
class IncrCacheFd:
    def __init__(self, mod, ids_t, cur_imap_t, len_t_m1):
        self.mod = mod
        self.ids_t = ids_t
        # idx map: keep this inside here!
        self.cur_imap_t = cur_imap_t
        # init states / histories
        _run_idx_t = torch.cat([len_t_m1*0, len_t_m1], dim=-1)  # [*, ??], origin idxes
        _run_mask_t = nnutils.constants(_run_idx_t.shape, 1., dtype=nnutils.DEFAULT_FLOAT)  # [*, ??]
        _tmp_pidx = torch.zeros_like(_run_idx_t)  # [*, ??]
        _tmp_pidx[..., 1] = 1  # [0,1]
        _run_posi_t = mod.incr_embed.forward_special(_tmp_pidx)  # [*, ??, D]
        self.run_idx_ts, self.run_mask_ts, self.run_posi_ts = [_run_idx_t], [_run_mask_t], [_run_posi_t]
        self.cur_ridx_base = 2  # current seq length
        self.cur_all_steps = [2]  # all steps
        self.cur_forward_ii = 0  # already forwarded stages
        # running cache
        self.posi_t = _run_posi_t  # [*, ??(ALL), D]
        self.stage_t = None  # [*, ??], stages of each one
        self.hid_t = None  # [*, ??, D], latest hidden layers before scoring!
        self.run_cache = {}
        # --
        # others
        self._arange_ts = {}
        # --

    def get_arange_t(self, bsize: int):
        ret = self._arange_ts.get(bsize)
        if ret is None:
            ret = nnutils.arange(bsize, dtype=nnutils.DEFAULT_INT).unsqueeze(-1)  # [*, 1]
            self._arange_ts[bsize] = ret
        return ret

    # [*, kk]
    def append(self, sel_idxes_t, sel_masks_t, sel_left_t, sel_right_t):
        # add to history
        self.run_idx_ts.append(sel_idxes_t)
        self.run_mask_ts.append(sel_masks_t)
        # update posi
        _cur_imap_t = self.cur_imap_t
        left_ridx, right_ridx = _cur_imap_t.gather(-1, sel_left_t), _cur_imap_t.gather(-1, sel_right_t)  # [*, kk]
        _arange_t = self.get_arange_t(sel_idxes_t.shape[0])  # [*, 1]
        left_es, right_es = self.posi_t[_arange_t, left_ridx], self.posi_t[_arange_t, right_ridx]  # [*, kk, D]
        new_posi_t = self.mod.incr_embed(left_es, right_es)  # [*, kk, D]
        self.run_posi_ts.append(new_posi_t)
        self.posi_t = torch.cat([self.posi_t, new_posi_t], -2)  # [*, ALL, D]
        # update imap
        _kk = sel_masks_t.shape[-1]
        _cur_imap_t.scatter_(-1, sel_idxes_t, torch.zeros_like(sel_idxes_t) + nnutils.arange(_kk) + self.cur_ridx_base)
        _cur_imap_t[..., 0] = 0  # reset for pad(0)
        # --
        self.cur_ridx_base += _kk
        self.cur_all_steps.append(_kk)
        # --

    # forward to the latest ones!
    def refresh_forward(self):
        _cur_ii, _trg_ii = self.cur_forward_ii, len(self.run_idx_ts)
        if _cur_ii < _trg_ii:
            _mod = self.mod
            # prepare inputs
            _idx_t = torch.cat(self.run_idx_ts[_cur_ii:], -1)  # [*, S]
            _mask_t = torch.cat(self.run_mask_ts[_cur_ii:], -1)  # [*, S]
            _steps = [self.cur_all_steps[ii] for ii in range(_cur_ii, _trg_ii)]  # [S]
            _cur_stage_t = nnutils.input_idx(sum([[_cur_ii+ii0]*vv for ii0, vv in enumerate(_steps)], []))  # [S]
            _cur_stage_t = _cur_stage_t + 0*_idx_t  # [*, S]
            # --
            # try to compress input!
            if _trg_ii - _cur_ii > 1:
                _new_sidxes_t, _new_mask_t = nnutils.mask2idx(_mask_t)  # [*, S']
                _orig_size, _new_size = _mask_t.shape[-1], _new_mask_t.shape[-1]
                if _new_size < _orig_size:  # need to compress
                    _arange_t = self.get_arange_t(_idx_t.shape[0])  # [*, 1]
                    # change input: [*, S']
                    _idx_t, _mask_t, _cur_stage_t = \
                        _idx_t[_arange_t, _new_sidxes_t], _new_mask_t, _cur_stage_t[_arange_t, _new_sidxes_t]
                    # change posi
                    _part0, _part1 = self.posi_t[..., :-_orig_size, :], self.posi_t[..., -_orig_size:, :]
                    self.posi_t = torch.cat([_part0, _part1[_arange_t, _new_sidxes_t]], -2)  # [*, ALL-S+S', D]
                    # change imap & cur_ridx_base
                    self.cur_ridx_base -= _orig_size  # ALL-(S'-S)
                    _cur_imap_t = self.cur_imap_t
                    _cur_imap_t.scatter_(-1, _idx_t, torch.zeros_like(_idx_t) + nnutils.arange(_new_size) + self.cur_ridx_base)
                    _cur_imap_t[..., 0] = 0  # reset for pad(0)
                    self.cur_ridx_base += _new_size
            # --
            _posi_t = self.posi_t[..., -_mask_t.shape[-1]:, :]  # [*, S, D]
            # to id
            _id_t = self.ids_t.gather(-1, _idx_t)  # [*, S]
            _id_t[_mask_t<=0.] = _mod.pad_token_id
            # causal mask
            _all_stage_t = _cur_stage_t if self.stage_t is None else torch.cat([self.stage_t, _cur_stage_t], -1)  # [*, ALL]
            _cmask_t = (_cur_stage_t.unsqueeze(-1) >= _all_stage_t.unsqueeze(-2)).to(nnutils.DEFAULT_FLOAT)  # [*, S, ALL]
            # forward model
            res = _mod.bert.forward_model(_id_t, extra_embeds=_posi_t, self_mask_k=_mask_t, self_mask_qk=_cmask_t, cache=self.run_cache)
            _cur_hid_t = res[0]
            # update
            self.stage_t = _all_stage_t
            self.hid_t = _cur_hid_t if self.hid_t is None else torch.cat([self.hid_t, _cur_hid_t], -2)  # [*, ALL, D]
            self.cur_forward_ii = _trg_ii
        # --

    # [*, ??], [*, elen]
    def score(self, left_t, right_t):
        _cur_imap_t = self.cur_imap_t
        # first map idxes to running idxes
        rleft_t, rright_t = _cur_imap_t.gather(-1, left_t), _cur_imap_t.gather(-1, right_t)  # [*, ??]
        # then gather and calculate
        _arange_t = nnutils.arange(rleft_t.shape[0], dtype=nnutils.DEFAULT_INT).unsqueeze(-1)  # [*, 1]
        hid_t0, hid_t1 = self.hid_t[_arange_t, rleft_t], self.hid_t[_arange_t, rright_t]  # [*, ??, D]
        mod = self.mod
        feat_t = mod.comb([hid_t0, hid_t1])  # [*, ??, D]
        out_t = mod.bert.lmhead(feat_t)  # [*, ??, V]
        return out_t

# --
# special position embeddings
class IncrEmbedConf(ZNodeConf):
    def __init__(self):
        self.esize = -1  # model size
        # --
        self.comb_method = ConfEntryChoices({"avg": None, "aff": ZAffineConf()}, "avg")
        # --

@node_reg(IncrEmbedConf)
class IncrEmbed(ZNode):
    def __init__(self, conf: IncrEmbedConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IncrEmbedConf = self.conf
        # --
        _dim = conf.esize
        self.special = torch.nn.Embedding(2, _dim)
        if conf.comb_method is None:  # simply average
            self.comb_f = lambda x,y: (x+y)/2
        elif isinstance(conf.comb_method, ZAffineConf):
            self.comb_node = ZAffineNode(conf.comb_method, isize=[_dim,_dim], osize=_dim)
            self.comb_f = lambda x,y: self.comb_node([x,y])
        else:
            raise NotImplementedError()
        # --
        self.to(nnutils.DEFAULT_DEVICE)
        # --

    def forward_special(self, idxes_t):
        ret = self.special(idxes_t)  # [*, D]
        return ret

    # [*, D]x2
    def forward(self, left_es, right_es):
        ret = self.comb_f(left_es, right_es)
        return ret  # [*, D]

# --
# b zgen/model/tasks/dec_inslm/helper_incr:
