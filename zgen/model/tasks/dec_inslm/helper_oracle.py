#

# helpers for getting oracles

__all__ = [
    "HelperOracle",
]

from typing import Dict
import torch
import math
from zgen.utils import Constants
from zgen.utils import nn as nnutils

class HelperOracle:
    def __init__(self, strategy: str, stat, vocab: Dict):
        self.strategy = strategy[:-1]
        self.rev = (strategy[-1] == "1")  # reverse (negate) it or not
        self._get_f = getattr(self, f"_get_{self.strategy}")
        self.stat = stat  # MyStat
        self.vocab = vocab  # word -> idx
        self._cache = None  # different according to different strategies
        # --
        # self.dtype = torch.float32
        self.dtype = nnutils.DEFAULT_FLOAT
        # --

    # [*, len]
    def get_oracle(self, idxes_t, masks_t):
        impt_scores = self._get_f(idxes_t, masks_t)
        if impt_scores is None:
            return None
        if self.rev:  # simply negate
            impt_scores *= -1
        # impt_scores[masks_t<=0] = Constants.REAL_PRAC_MIN
        impt_scores[masks_t<=0] = float('-inf')
        return impt_scores

    # l2r: [0, -1, -2, -3, ...]
    def _get_l2r(self, idxes_t, masks_t):
        ret = torch.zeros_like(masks_t, dtype=self.dtype) - nnutils.arange(masks_t.shape[-1], dtype=self.dtype)
        return ret

    # freq: log(1+freq)
    def _get_freq(self, idxes_t, masks_t):
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
    def _get_tfidf(self, idxes_t, masks_t):
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
    def _get_random(self, idxes_t, masks_t):
        ret = (nnutils.rand(idxes_t.shape, dtype=torch.float) + 1e-10).log().to(self.dtype)
        return ret

    # uniform
    def _get_uniform(self, idxes_t, masks_t):
        ret = torch.zeros_like(masks_t, dtype=self.dtype)
        return ret

    # balance tree: ...
    def _get_btree(self, idxes_t, masks_t):
        if self._cache is None:
            _MAX_LEN = 512  # todo(note): currently this should be enough!
            _tmpvs = []
            for ii in range(_MAX_LEN):
                _tmpvs.append(HelperOracle._impt_btree(ii))
            # note: here 0 does not matter since will be masked out later
            _vv, _ = nnutils.go_batch_2d(_tmpvs, 0., dtype=self.dtype)  # [MAX_LEN, MAX_LEN]
            self._cache = _vv
        else:
            _vv = self._cache
        _len = idxes_t.shape[-1]
        idxes_t = masks_t.sum(-1).to(nnutils.DEFAULT_INT)  # [*]
        ret = _vv[idxes_t][..., :_len]  # [*, _len]
        return ret

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
    def _get_model(self, idxes_t, masks_t):
        return None  # currently do not know!!

    # real bt (distance to span center)
    def _get_bt(self, idxes_t, masks_t):
        return None  # currently do not know, to compute on the fly

# --
# b zgen/model/tasks/dec_inslm/helper_oracle:74
