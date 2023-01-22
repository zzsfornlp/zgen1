#

# helpers for getting oracles

__all__ = [
    "HelperOracleConf", "HelperOracle", "PsHelperConf", "PsHelper",
]

from typing import Dict
import torch
import torch.nn.functional as F
import math
import numpy as np
from zgen.utils import Constants, Conf, ScheduledValue, SVConf, zwarn, zlog
from zgen.utils import nn as nnutils

class HelperOracleConf(Conf):
    def __init__(self):
        # some confs for specific strategies
        self.fbin_alpha = 1.0  # smooth counts?
        self.fbin_bin = 3  # how many bins to put
        self.fbin_bin2score = [-2., -1., 0.]  # by default rare first
        self.eps_add_ends = True  # add ends to both ends
        # --

class HelperOracle:
    def __init__(self, strategy: str, stat, vocab: Dict, conf: HelperOracleConf):
        self.conf = conf
        # --
        if str.isdigit(strategy[-1]):  # ignore last digit for indicating Reverse or not
            strategy = strategy[:-1]
        self.strategy = strategy
        self._get_f = getattr(self, f"_get_{self.strategy}")
        self.stat = stat  # MyStat
        self.vocab = vocab  # word -> idx
        self._cache = None  # different according to different strategies
        # --
        # self.dtype = torch.float32
        self.dtype = nnutils.DEFAULT_FLOAT
        # --

    # [*, len]
    def get_oracle(self, idxes_t, masks_t, med):
        impt_scores = self._get_f(idxes_t, masks_t, med)
        return impt_scores

    # l2r: [0, -1, -2, -3, ...]
    def _get_l2r(self, idxes_t, masks_t, med):
        ret = torch.zeros_like(masks_t, dtype=self.dtype) - nnutils.arange(masks_t.shape[-1], dtype=self.dtype)
        return ret

    # freq: log(1+freq)
    def _get_freq(self, idxes_t, masks_t, med):
        if self._cache is None:
            _freqs = [0.] * len(self.vocab)
            for ww, ii in self.vocab.items():
                _freqs[ii] = math.log(1+self.stat.count_tf.get(ww, 0))
            _freqs_t = nnutils.input_tensor(_freqs, dtype=self.dtype).log_softmax(-1)
            self._cache = _freqs_t
        else:
            _freqs_t = self._cache
        # --
        ret = _freqs_t[idxes_t]
        return ret

    # tfidf: tf * idf
    def _get_tfidf(self, idxes_t, masks_t, med):
        _vlen = len(self.vocab)
        if self._cache is None:
            _default_idf = self.stat.get_idf("")
            _idfs = [_default_idf] * _vlen
            for ww, ii in self.vocab.items():
                _idfs[ii] = self.stat.get_idf(ww)
            _idfs_t = nnutils.input_tensor(_idfs, dtype=self.dtype)
            self._cache = _idfs_t
        else:
            _idfs_t = self._cache
        # --
        cur_idf = _idfs_t[idxes_t]
        _counts = nnutils.zeros(list(idxes_t.shape)[:-1]+[_vlen], dtype=nnutils.DEFAULT_INT)
        _counts.scatter_add_(-1, idxes_t, torch.ones_like(idxes_t))  # add counts
        cur_tf = _counts.gather(-1, idxes_t).to(self.dtype) / masks_t.sum(-1, keepdims=True)  # gather and get freqs
        ret = cur_tf * cur_idf
        return ret

    # random
    def _get_random(self, idxes_t, masks_t, med):
        ret = (nnutils.rand(idxes_t.shape, dtype=torch.float) + 1e-10).log().to(self.dtype)
        return ret

    # uniform
    def _get_uniform(self, idxes_t, masks_t, med):
        ret = torch.zeros_like(masks_t, dtype=self.dtype)
        return ret

    # deprecated by (on-the-fly) bt!!
    # # balance tree: ...
    # def _get_btree(self, idxes_t, masks_t, med):
    #     if self._cache is None:
    #         _MAX_LEN = 512  # todo(note): currently this should be enough!
    #         _tmpvs = []
    #         for ii in range(_MAX_LEN):
    #             _tmpvs.append(HelperOracle._impt_btree(ii))
    #         # note: here 0 does not matter since will be masked out later
    #         _vv, _ = nnutils.go_batch_2d(_tmpvs, 0., dtype=self.dtype)  # [MAX_LEN, MAX_LEN]
    #         self._cache = _vv
    #     else:
    #         _vv = self._cache
    #     _len = idxes_t.shape[-1]
    #     idxes_t = masks_t.sum(-1).to(nnutils.DEFAULT_INT)  # [*]
    #     ret = _vv[idxes_t][..., :_len]  # [*, _len]
    #     return ret

    # helper for btree
    @staticmethod
    def _impt_btree(_length):  # balanced tree
        # --
        def _rec(_len: int, _v):
            if _len <= 0:
                return []
            elif _len == 1:
                return [_v]
            _mid = (_len - 1) // 2
            _left, _right = _rec(_mid, _v-1), _rec(_len - 1 - _mid, _v-1)
            return _left + [_v] + _right
        # --
        ret = _rec(_length, 0)
        return ret

    # model score based
    def _get_model(self, idxes_t, masks_t, med):
        return None  # currently do not know!!

    # real bt (distance to span center)
    def _get_bt(self, idxes_t, masks_t, med):
        return None  # currently do not know, to compute on the fly

    # read ps scores from aux
    def _get_ps(self, idxes_t, masks_t, med):
        _eps_add_ends = self.conf.eps_add_ends
        _lastd = 1 if len(med.ibatch.insts)==0 else med.ibatch.insts[0]._aux.shape[-1]
        # --
        _shape = list(idxes_t.shape)
        _shape = _shape + [_lastd]  # [*, L, D]
        arr = np.full(_shape, fill_value=0.)  # [*, L, D]
        for ii, inst in enumerate(med.ibatch.insts):
            aux = inst._aux  # [len, D]
            if _eps_add_ends:
                assert len(aux)+2 == len(inst)
                arr[ii, 1:(1+len(aux))] = aux
            else:
                assert len(aux) == len(inst)
                arr[ii, :len(aux)] = aux
        ret = nnutils.input_real(arr)
        return ret  # [*, L, D]

    # read external scores from aux
    def _get_es(self, idxes_t, masks_t, med):
        _eps_add_ends = self.conf.eps_add_ends
        _shape = list(idxes_t.shape)
        arr = np.full(_shape, fill_value=0.)  # [*, L]
        for ii, inst in enumerate(med.ibatch.insts):
            aux = inst._aux  # [L]
            if _eps_add_ends:
                assert len(aux)+2 == len(inst)
                arr[ii, 1:(1+len(aux))] = aux
            else:
                assert len(aux) == len(inst)
                arr[ii, :len(aux)] = aux
        ret = nnutils.input_real(arr)
        return ret

    # get by freq-bin
    def _get_fbin(self, idxes_t, masks_t, med):
        if self._cache is None:
            _bin, _bin2score = self.conf.fbin_bin, self.conf.fbin_bin2score
            assert len(_bin2score) >= _bin
            # --
            _lenV = len(self.vocab)
            full_vocab_counts = [0] * _lenV  # [V]
            for k, v in self.stat.count_tf.items():
                ii = self.vocab.get(k, None)
                if ii is None:
                    zwarn(f"Cannot find key: {k}")
                    continue
                full_vocab_counts[ii] = v
            # sort by freq: [V]
            _sort_idxes = np.argsort(full_vocab_counts)[::-1]
            _sort_values = [full_vocab_counts[z] for z in _sort_idxes]
            _arr_values = np.asarray(_sort_values) ** self.conf.fbin_alpha
            # _sort_percV = (np.arange(_lenV) + 1) / _lenV
            _sort_percF = np.cumsum(_arr_values) / np.sum(_arr_values)
            _arr_scores = np.full(_lenV, fill_value=float(min(_bin2score)))
            for ii, vv in zip(_sort_idxes, _sort_percF):
                _arr_scores[ii] = _bin2score[min(_bin-1, int(vv * _bin))]  # which bin to put and which score?
            _idx2score = nnutils.input_tensor(_arr_scores, dtype=self.dtype)
            # --
            # printing
            _arr_counts = np.asarray(full_vocab_counts)
            _all_count = _arr_counts.sum().item()
            _print_res = []
            for i,v in enumerate(_bin2score):
                _rr = f"{(_idx2score==v).sum().item()}({(_arr_counts*(_idx2score==v).cpu().numpy()).sum() / _all_count:.4f})"
                _print_res.append(_rr)
            zlog(f"Fbin partition = {_print_res}")
            # --
            self._cache = _idx2score
        else:
            _idx2score = self._cache
        # --
        ret = _idx2score[idxes_t]
        return ret

# --
# extra helper for pair-score based one

class PsHelperConf(Conf):
    def __init__(self):
        # for score (for ps)
        # self.is_ps = False
        self.app_winr = 10  # applied window range, should be <=rr
        # for filter gap
        self.filter_gap = -1.  # activated if >0.
        # for rank
        self.do_rank = False  # norm scores to [0, -1, -2, ...]
        # for center
        self.do_center = False  # do centering
        self.center_only_posi = True  # fall back to only positional center
        self.center_range = 2.  # range of centering!
        self.center_alpha = 1.  # further divide dist by this!
        self.center_mixu = SVConf.direct_conf(val=0.)  # mix uniform?
        # center ordering method
        self.center_ordering = 'mid'  # l2r/r2l/mid
        # --

class PsHelper:
    def __init__(self, conf: PsHelperConf):
        self.conf = conf
        self.center_mixu = ScheduledValue(f"center_mixu", conf.center_mixu)
        # --
        # center ordering mode
        self._co_l2r, self._co_r2l, self._co_mid = [conf.center_ordering==z for z in ['l2r','r2l','mid']]
        # --

    # [..., L, R+1+R], [..., L];; [*, k], [*, k], [*, L]
    def score(self, input_score, trg_mask, left_t, right_t, stage_t):
        conf = self.conf
        _PEN = -100.  # large penalty
        # --
        with nnutils.no_grad_env():
            # common ones
            # scoring
            # if conf.is_ps:
            if len(input_score.shape) > len(trg_mask.shape):  # need to reduce as ps!
                _rr, _ll = (input_score.shape[-1]-1)//2, trg_mask.shape[-1]
                _tmp_mask = F.pad(trg_mask, [_rr, _rr], value=0.).unsqueeze(-2)  # [..., 1, R+L+R]
                _tm_shape = list(trg_mask.shape)  # [..., L]
                _aa = (nnutils.arange(_rr*2+1) + nnutils.arange(_ll).unsqueeze(-1))\
                    .view([1]*(len(_tm_shape)-1) + [_ll, 2*_rr+1]).expand(_tm_shape[:-1]+[-1,-1])  # [..., L, R+1+R]
                _mm = _tmp_mask.expand([-1]*(len(_tm_shape)-1) + [_ll, -1]).gather(-1, _aa)  # [..., L, R+1+R]
                # --
                # specific mask for scoring winr!
                _mm[..., _rr] = 0.  # first exclude self
                _mm[..., :max(0, _rr-conf.app_winr)] = 0.
                _mm[..., (_rr+conf.app_winr+1):] = 0.
                _mm *= trg_mask.unsqueeze(-1)  # easier to see and debug!
                # --
                _sum = (input_score * _mm).sum(-1)  # [..., L]
                _div = _mm.sum(-1).clamp(min=1.)  # [..., L]
                _score0 = _sum / _div  # [..., L]
            else:
                _score0 = input_score  # [..., L]
            # --
            # clone mask since we might need to change it!
            _cur_mask = trg_mask.clone()  # [..., L]
            # filter gap
            if conf.filter_gap > 0:
                _max_score0 = (_score0 + _PEN * (1. - _cur_mask)).max(-1, keepdims=True)[0]  # [..., 1]
                _oor = (_score0+conf.filter_gap) < _max_score0  # [..., L], out of range?
                _cur_mask[_oor] = 0.  # [..., L]
            # --
            # centering
            if conf.do_center:
                _cur_count = _cur_mask.sum(-1, keepdims=True)  # [..., 1]
                if conf.center_only_posi:  # take position
                    _posi = _cur_mask.cumsum(-1)  # [..., L]
                else:
                    _posi = (_score0 + _PEN * (1. - _cur_mask)).softmax(-1).cumsum(-1) * _cur_count  # [..., L]
                # --
                _posi[_cur_mask<=0.] = float(_cur_mask.shape[-1]) + 10.  # make it large temporally
                _min_posi = _posi.min(-1, keepdims=True)[0]  # [..., 1]
                _max_posi = _cur_count  # [..., 1]
                if self._co_l2r:
                    _center = _min_posi
                elif self._co_r2l:
                    _center = _max_posi
                elif self._co_mid:
                    _center = (_min_posi + _max_posi) / 2
                else:
                    raise NotImplementedError()
                # --
                _posi[_cur_mask<=0.] = _PEN  # finally make it distant enough
                _dist = (_posi - _center).abs()  # [..., L]
                _dist -= _dist.min(-1, keepdims=True)[0]  # [..., L], make min as 0.
                # --
                _cur_mask[_dist > conf.center_range] *= 0.  # [..., L]
                _score0 = (-_dist) / conf.center_alpha  # [..., L]
            # --
            # convert raw scores to rank
            if conf.do_rank:
                _tmp_score0 = _score0 + _PEN * (1. - _cur_mask)  # exclude invalid ones!
                # _tmp_score1 = _tmp_score0.sort(-1, descending=True)[0]  # [..., L]
                _TOPK = 40  # note: no need for others!
                if _tmp_score0.shape[-1] <= _TOPK:
                    _tmp_score1 = _tmp_score0
                else:
                    _tmp_score1 = _tmp_score0.topk(_TOPK, dim=-1, sorted=False)[0]
                # [..., L], put things into rankings: {0, -1, -2, ...}
                _tmp_score2 = - (_tmp_score0.unsqueeze(-1) < _tmp_score1.unsqueeze(-2)).to(
                    nnutils.DEFAULT_FLOAT).sum(-1)
                _score0 = _tmp_score2
            # --
            # mix uniform?
            _mixu = self.center_mixu.value
            if _mixu > 0.:
                _uniform_weight = _cur_mask / _cur_mask.sum(-1, keepdims=True).clamp(min=1.)  # [..., L]
                _mix_weight = (1. - _mixu) * _score0.softmax(-1) + _mixu * _uniform_weight  # [..., L]
                _score0 = (_mix_weight + 1e-6).log()  # [..., L]
            # --
            # final apply mask to _score0
            ret = _score0 + 0. * _cur_mask  # [..., L], expand if needed
            ret[_cur_mask <= 0.] = _PEN
            return ret  # [..., L]
        # --

# --
# b zgen/model/tasks/dec_ilm/help_oracle:128
