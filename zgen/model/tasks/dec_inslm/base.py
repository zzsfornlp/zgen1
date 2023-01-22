#

# base decoder

__all__ = [
    "ZTaskInslm", "ZTaskInslmConf", "ZDecoderInslmConf", "ZDecoderInslm",
]

from typing import List
import numpy as np
from copy import deepcopy
from collections import Counter, OrderedDict
import torch
import torch.nn.functional as F
from zgen.utils import ResultRecord, MathHelper, zwarn, zlog, ScheduledValue, SVConf, MyStat, zglob1, ConfEntryChoices
from zgen.utils import nn as nnutils
from ...core import *
from ..base import *
from ..helper import eval_inslm
from .helper_oracle import *

# --

class ZTaskInslmConf(ZTaskBaseTConf):
    def __init__(self):
        super().__init__()
        self.name = "inslm"
        from .model1 import ZDecoderInslm1Conf
        from .model2 import ZDecoderInslm2Conf
        self.inslm_conf = ConfEntryChoices({'v1': ZDecoderInslm1Conf(), 'v2': ZDecoderInslm2Conf()}, "v1")
        # --

    def build_task(self):
        return ZTaskInslm(self)

class ZTaskInslm(ZTaskBaseT):
    def __init__(self, conf: ZTaskInslmConf):
        super().__init__(conf)
        # --

    def build_mod(self, model):
        return self.conf.inslm_conf.make_node(self, model)

    def eval_insts(self, gold_insts, pred_insts, quite=False):
        return eval_inslm(self.name, gold_insts, pred_insts, quite)
# --

class ZDecoderInslmConf(ZModBaseTConf):
    def __init__(self):
        super().__init__()
        # --
        self.loss_inslm = 1.
        self.comb = ZAffineConf.direct_conf(act='elu')
        self.noi_name = '[unused2]'
        # roll-in and oracle
        self.mrate_low = SVConf.direct_conf(val=0.)
        self.mrate_high = SVConf.direct_conf(val=1.)
        self.rollin_strategy = "random0"  # l2r/freq=(log(1+f))/tfidf/random/btree + R?
        self.oracle_strategy = "uniform0"  # l2r/freq/tfidf/uniform/btree/model + R?
        self.oracle_tau = SVConf.direct_conf(val=1.)
        self.oracle_mixu = SVConf.direct_conf(val=0.)  # mix uniform?
        self.oracle_model_detach = True  # detach for p?
        self.stat_file = ""
        # how to weight loss?
        self.loss_sample_one = False
        # how to do forced decoding
        self.fd_oracle_strategy = "random0"  # ... + R?
        # --

@node_reg(ZDecoderInslmConf)
class ZDecoderInslm(ZModBaseT):
    def __init__(self, conf: ZDecoderInslmConf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZDecoderInslmConf = self.conf
        # --
        # some specific ones
        self.noi_token_id = self.tokenizer.convert_tokens_to_ids([conf.noi_name])[0]
        assert self.tokenizer.convert_ids_to_tokens([self.noi_token_id]) == [conf.noi_name]
        zlog(f"Use {conf.noi_name}([id={self.noi_token_id}]) for NOI!")
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        _hid_dim = self.bert.hidden_size
        # extra feature combiner
        self.comb = ZAffineNode(conf.comb, isize=[_hid_dim, _hid_dim], osize=_hid_dim)
        # --
        self.mrate_low = ScheduledValue(f"mrate_low", conf.mrate_low)
        self.mrate_high = ScheduledValue(f"mrate_high", conf.mrate_high)
        self.oracle_tau = ScheduledValue(f"oracle_tau", conf.oracle_tau)
        self.oracle_mixu = ScheduledValue(f"oracle_mixu", conf.oracle_mixu)
        self.rollin_specs = self._check_special_strategies(conf.rollin_strategy)
        self.oracle_specs = self._check_special_strategies(conf.oracle_strategy)
        self.fd_specs = self._check_special_strategies(conf.fd_oracle_strategy)
        # --
        # oracle helpers
        if conf.stat_file:
            self.stat = MyStat.create_from_file(zglob1(conf.stat_file, check_iter=10))
        else:
            self.stat = None
        _vocab = self.tokenizer.get_vocab()
        self.rollin_helper = HelperOracle(conf.rollin_strategy, self.stat, _vocab)
        self.rollin_noise = 0.1 if conf.rollin_strategy.startswith("btree") else 0.  # use noise to break tie
        self.oracle_helper = HelperOracle(conf.oracle_strategy, self.stat, _vocab)
        self.fd_helper = HelperOracle(conf.fd_oracle_strategy, self.stat, _vocab)
        # --
        zwarn("This model is deprecated by ILM!!")  # note: deprecated model!
        exit(1)
        # --

    def _get_scheduled_values(self):
        return OrderedDict([("_mrate_low", self.mrate_low), ("_mrate_high", self.mrate_high),
                            ("_oracle_tau", self.oracle_tau), ("_oracle_mixu", self.oracle_mixu)])

    def _check_special_strategies(self, name: str):
        stra_alpha = -1 if (name[-1] == "1") else 1
        name = name[:-1]
        stra_model, stra_bt = [name==z for z in ["model", "bt"]]
        return stra_model, stra_bt, stra_alpha

    # --
    # useful common ones

    # use mask2idx to select and compress!
    def _select_and_compress(self, masks_t, *values, pad=None):
        _tmp_idxes, ret_masks_t = nnutils.mask2idx(masks_t)  # [*, k']
        ret_values = [z.gather(-1, _tmp_idxes) for z in values]
        if pad is not None:
            _mm = (ret_masks_t <= 0.)
            for v in ret_values:
                v[_mm] = pad
        return ret_masks_t, *ret_values

    # add length idxes; eg: [0, 1, 3, 4, x, x] -> [0, 1, 3, 4, L, x, x]
    def _extend_length_idxes(self, _idxes_t, _masks_t, _len_t, pad=0):
        # expand another idx as the seq length (as sentinel)
        _extend_idxes_t = F.pad(_idxes_t, [0,1], value=pad)  # [*, klen+1]
        _extend_idxes_t.scatter_(-1, _masks_t.long().sum(-1, keepdims=True), _len_t)  # [*, klen+1]
        return _extend_idxes_t

    # valid2seg; eg: [1, 0, 1, 1, 0, 1] -> [[0, 2, 3, 5], [2, 3, 5, 6]], ...
    def _valid2seg(self, input_t, len_t, closed_t=None, ret_seg=True):
        # obtain blanks
        one_left_t0, one_vmasks_t0 = nnutils.mask2idx(input_t)  # [*, k0], current left boundaries
        one_extend_t0 = self._extend_length_idxes(one_left_t0, one_vmasks_t0, len_t)  # [*, k0+1]
        one_right_t0 = one_extend_t0[..., 1:]  # [*, k0], right boundaries
        if closed_t is None:
            one_vmasks_t, one_left_t, one_right_t = one_vmasks_t0, one_left_t0, one_right_t0
        else:  # another filtering!
            # exclude closed ones, remainings are to-fill blanks!
            one_vmasks_t0 *= (1. - closed_t.gather(-1, one_left_t0))  # [*, k0]
            one_vmasks_t, one_left_t, one_right_t = \
                self._select_and_compress(one_vmasks_t0, one_left_t0, one_right_t0, pad=0)  # [*, k]
        # obtain segments?
        if ret_seg:
            if nnutils.is_zero_shape(one_left_t):  # []
                _max_seg_size = 1
            else:
                _max_seg_size = max(1, int((one_right_t - one_left_t).max().item()))
            seg_idxes_t = nnutils.arange(_max_seg_size) + one_left_t.unsqueeze(-1)  # [*, k, S]
            seg_mask_t = (seg_idxes_t < one_right_t.unsqueeze(-1)).to(nnutils.DEFAULT_FLOAT)  # [*, klen, S], valid ones
            seg_idxes_t[seg_mask_t <= 0.] = 0  # make it safe with 0
        else:
            seg_idxes_t, seg_mask_t = None, None
        # --
        return one_left_t, one_right_t, one_vmasks_t, seg_idxes_t, seg_mask_t

    # stage_t to closed_t; [*, len]
    def _stage2close(self, cur_stage: int, stage_t, len_t):
        ok1 = (stage_t<cur_stage) & (stage_t>=0)  # the first one is ok
        ok2 = F.pad(ok1[..., 1:], [0,1], value=True)  # the second one is ok
        both_ok = ok1 & ok2
        # note: also exclude the last one!!
        ret = (both_ok | (nnutils.arange(both_ok.shape[-1]) >= (len_t-1))).to(nnutils.DEFAULT_FLOAT)
        return ret  # [*, len]

    # common routine for gather loss
    # [*, K, C], [*, K, C], object, [*, L], [*, K, C]
    # note: trg_mask_t[...,0] must mean NOI and should be 1. accordingly
    def _collect_loss(self, nll_t, trg_mask_t, oracle_specs, oracle_scores_t, seg_idxes=None):
        _seg_shape = list(nll_t.shape)  # [*, K, C]
        # use different oracles
        stra_model, stra_bt, stra_alpha = oracle_specs
        if stra_model:  # model-based
            _oracle_t = nll_t.detach() if self.conf.oracle_model_detach else nll_t
            _oracle_t = _oracle_t * (-stra_alpha)  # -nll = ll
        elif stra_bt:  # on-the-fly balance-tree
            # _oracle_t = - (((trg_mask_t.sum(-1, keepdims=True)+1) // 2) - nnutils.arange(_seg_shape[-1])).abs()
            # note: does not matter for the NOI since it adds 0
            _arange_t = nnutils.arange(_seg_shape[-1], dtype=nnutils.DEFAULT_INT)  # [C]
            _center = ((_arange_t * trg_mask_t).sum(-1) // trg_mask_t.sum(-1).clamp(min=1)).unsqueeze(-1)  # [*, k, 1]
            _oracle_t = - (_center - _arange_t).abs()  # [*, k, elen], distance to center
            _oracle_t = _oracle_t * stra_alpha
        else:
            if seg_idxes is None:  # in this case, C==L
                _oracle_t = oracle_scores_t.unsqueeze(-2)  # [*, 1, L]
            else:  # further select
                _oracle_t = oracle_scores_t.gather(-1, seg_idxes.view(_seg_shape[:-2]+[-1])).view(_seg_shape)
        # add -inf
        _extra_ones = torch.zeros_like(trg_mask_t)
        _extra_ones[trg_mask_t <= 0.] = float('-inf')
        _extra_ones[trg_mask_t.sum(-1) == 0.] = 0.  # avoid all -inf rows
        _oracle_t = _oracle_t + _extra_ones
        # scale by temp
        _tau = self.oracle_tau.value
        if _tau == 1.:  # [*, K, C]
            trg_weight_t = _oracle_t.softmax(-1)
        else:  # [*, klen, S]
            trg_weight_t = (_oracle_t.log_softmax(-1) / _tau).softmax(-1)  # first norm to avoid overflow
        # mix uniform
        _mixu = max(0., min(1., self.oracle_mixu.value))
        if _mixu > 0.:
            _uniform_weight = trg_mask_t / trg_mask_t.sum(-1, keepdims=True).clamp(min=1.)  # [*, klen, S]
            trg_weight_t = (1.-_mixu) * trg_weight_t + _mixu * _uniform_weight
        # sample one?
        if self.conf.loss_sample_one:
            # note: log(0) becomes -inf, which will surely not be sampled if there are non -inf!
            _, _sample_idxes = nnutils.category_sample(trg_weight_t.log())  # [*, K, 1]
            trg_weight_t = torch.zeros_like(trg_weight_t)
            trg_weight_t.scatter_(-1, _sample_idxes, 1.)  # [*, K, C]
        # --
        trg_weight_t = trg_weight_t * trg_mask_t  # remember the mask!!
        final_loss_t = (nll_t * trg_weight_t).sum(-1)  # [*, K]
        return final_loss_t  # [*, K]

# --
# b zgen/model/tasks/dec_inslm/base:?
