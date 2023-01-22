#

# seq lm (with causal l2r mask)

__all__ = [
    "ZTaskSlm", "ZTaskSlmConf", "ZDecoderSlmConf", "ZDecoderSlm",
]

from typing import List
import numpy as np
from copy import deepcopy
from collections import Counter, OrderedDict
import torch
from zgen.utils import nn as nnutils
from zgen.utils import zlog, ScheduledValue, SVConf
from ..core import *
from .base import *
from .helper import eval_inslm

# --

class ZTaskSlmConf(ZTaskBaseTConf):
    def __init__(self):
        super().__init__()
        self.name = "slm"
        self.slm_conf = ZDecoderSlmConf()
        self.test_with_fd = False
        # --

    def build_task(self):
        return ZTaskSlm(self)

class ZTaskSlm(ZTaskBaseT):
    def __init__(self, conf: ZTaskSlmConf):
        super().__init__(conf)
        # --

    def build_mod(self, model):
        return self.conf.slm_conf.make_node(self, model)

    def eval_insts(self, gold_insts, pred_insts, quite=False):
        if self.conf.test_with_fd:
            return eval_inslm(self.name, gold_insts, pred_insts, quite=quite)
        else:
            return super().eval_insts(gold_insts, pred_insts, quite=quite)
# --

class ZDecoderSlmConf(ZModBaseTConf):
    def __init__(self):
        super().__init__()
        # --
        self.loss_slm = 1.
        self.rev_seq = False  # r2l
        # --
        self.train_curriculum = SVConf.direct_conf(val=1., min_val=0., max_val=1.)  # from 0 to 1, by default always 1
        # testing
        self.test_beam_size = 4
        self.test_max_len = 500
        self.test_max_step = 100000
        self.test_max_ratio = 1.5  # max ratio to src if there are
        self.test_eos_penalty = 0.
        self.test_len_reward = 0.  # encourage lengthy ones, depend on specific tasks!
        self.test_len_penalty = 0.6  # length norm
        # test with sample
        self.test_do_sample = False
        self.test_sample_topk = 0
        self.test_sample_topp = 0.
        self.test_sample_max_bs = 128  # sample how many at each time?
        # test output
        self.test_output_all = False
        # --

@node_reg(ZDecoderSlmConf)
class ZDecoderSlm(ZModBaseT):
    def __init__(self, conf: ZDecoderSlmConf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZDecoderSlmConf = self.conf
        # --
        self.train_curriculum = ScheduledValue(f"train_curriculum", conf.train_curriculum)
        # --

    def _get_scheduled_values(self):
        return OrderedDict([("_train_curriculum", self.train_curriculum),])

    def set_test_len_reward(self, r: float):
        old_p = self.conf.test_len_reward
        self.conf.test_len_reward = r
        zlog(f"[!!] Set LEN-REWARD from {old_p} to {self.conf.test_len_reward}")
        return self.conf.test_len_reward
        # --

    def set_test_sample_p(self, p: float):
        if p<=0.:
            self.conf.test_do_sample = False
        else:
            self.conf.test_do_sample = True
        old_p = self.conf.test_sample_topp
        self.conf.test_sample_topp = p
        zlog(f"[!!] Set Sample-TopP from {old_p} to {self.conf.test_sample_topp}")
        return self.conf.test_sample_topp
        # --

    def set_test_eos_penalty(self, p: float):
        old_p = self.conf.test_eos_penalty
        self.conf.test_eos_penalty = p
        zlog(f"[!!] Set EOS-penalty from {old_p} to {self.conf.test_eos_penalty}")
        return self.conf.test_eos_penalty
        # --

    def set_test_confs(self, sample_p: float = None, eos_p: float = None, *args, **kwargs):
        if sample_p is not None:
            self.set_test_sample_p(sample_p)
        if eos_p is not None:
            self.set_test_eos_penalty(eos_p)
        # --

    def calc_output(self, med, idxes_t, masks_t):
        # --
        cross_t, cross_mask_k, _ = self.prepare_search(med, 1, 1, 1.)
        # --
        _arange_t = nnutils.arange(masks_t.shape[-1]).view([1]*(len(idxes_t.shape)-1)+[-1]).expand_as(idxes_t)  # [*, _len]
        causal_masks_t = (_arange_t.unsqueeze(-2)<=_arange_t.unsqueeze(-1)).to(nnutils.DEFAULT_FLOAT)  # [*, _len, _len]
        hid_t = self.forward_bert_model(input_ids=idxes_t, self_mask_k=masks_t, self_mask_qk=causal_masks_t,
                                        cross_t=cross_t, cross_mask_k=cross_mask_k)[0]  # [*, _len, D]
        out_t = self.bert.lmhead(hid_t)  # [*, _len, V]
        return out_t

    def calc_nll(self, med: ZMediator, label_smoothing: float):
        _rev_seq = self.conf.rev_seq
        # first prepare input!
        IDX_PAD = self.tokenizer.pad_token_id
        ids = self.get_inst_idxes(med.ibatch.insts)
        if _rev_seq:
            ids = [list(reversed(ones)) for ones in ids]
        idxes_t, masks_t = nnutils.go_batch_2d(ids, IDX_PAD)  # [*, elen]
        # then forward
        out_t = self.calc_output(med, idxes_t, masks_t)  # [*, elen, V]
        nll_t = nnutils.loss_nll(out_t[..., :-1, :], idxes_t[..., 1:], label_smoothing=label_smoothing)  # [*, klen-1]
        return nll_t, masks_t

    def do_loss(self, med: ZMediator, *args, **kwargs):
        nll_t, masks_t = self.calc_nll(med, self.conf.label_smoothing)
        # curriculum
        curriculum = self.train_curriculum.value
        # loss
        final_mask_t = masks_t[..., 1:]  # [*, klen-1]
        if curriculum != 1.0:
            arange_t = nnutils.get_arange_t(final_mask_t.shape[-1], 0)  # [klen-1]
            _mul = (curriculum ** arange_t).to(nnutils.DEFAULT_FLOAT)
            final_mask_t = final_mask_t * _mul
        _loss_item = LossHelper.compile_leaf_loss(
            f'{self.name}_nll', (nll_t*final_mask_t).sum(), final_mask_t.sum(), loss_lambda=self.conf.loss_slm)
        # --
        ret = LossHelper.combine_multiple_losses([_loss_item])
        return ret, {}

    def do_predict(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderSlmConf = self.conf
        # --
        if self.ztask.conf.test_with_fd:
            return self._test_fd(med, *args, **kwargs)
        else:
            if conf.test_do_sample and conf.test_beam_size > conf.test_sample_max_bs:
                orig_bs = conf.test_beam_size
                conf.test_beam_size = conf.test_sample_max_bs
                ret = {}
                for _ in range((orig_bs+conf.test_sample_max_bs-1)//conf.test_sample_max_bs):
                    ret = self._test_od(med, *args, **kwargs)  # run it multiple times
                conf.test_beam_size = orig_bs
                return ret
            else:
                return self._test_od(med, *args, **kwargs)
        # --

    def _test_fd(self, med: ZMediator, *args, **kwargs):
        _rev_seq = self.conf.rev_seq
        nll_t, masks_t = self.calc_nll(med, 0.)
        logprob_t = - nll_t
        # assign
        arr_logprob = nnutils.get_value(logprob_t)
        for bidx, inst in enumerate(med.ibatch.insts):
            _len = len(inst)
            cur_logprobs = arr_logprob[bidx, :_len - 1].tolist()
            inst.info['slm_info'] = (
            list(range(len(inst))), [0.] + cur_logprobs[:-1] + [0.], [0.] * (_len - 1) + [cur_logprobs[-1]])
            if _rev_seq:
                inst.info['slm_info'] = tuple([list(reversed(z)) for z in inst.info['slm_info']])
        # --
        return {}

    # do beam search
    def _test_od(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderSlmConf = self.conf
        # --
        # prepare
        bsize = len(med.ibatch.insts)
        beam_size = conf.test_beam_size
        cross_t, cross_mask_k, max_len_t = self.prepare_search(med, beam_size, conf.test_max_len, conf.test_max_ratio)
        # --
        # run search
        bert = self.bert
        svoc = self.ztask.special_vocab
        pad_id, bos_id, eos_id = svoc.pad_token_id, svoc.cls_token_id, svoc.sep_token_id
        if conf.rev_seq:  # simply reverse these symbols!
            bos_id, eos_id = eos_id, bos_id
        _NEG_INF = -10000.
        _PRIZE = 100.
        eos_penalty = conf.test_eos_penalty
        # --
        # prepare status
        cur_id_t = nnutils.constants([bsize*beam_size, 1], value=bos_id, dtype=nnutils.DEFAULT_INT)  # [bs*B, 1]
        cur_mask_t = nnutils.constants([bsize*beam_size, 1], value=1., dtype=nnutils.DEFAULT_FLOAT)  # [bs*B, 1]
        cur_finished_t = (cur_mask_t<=0.).squeeze(-1)  # [bs*B]
        run_cache = {}
        # prepare special masks: [0, -inf, ...]
        _beam_mask_v = nnutils.constants([beam_size], value=_NEG_INF)  # [B]
        _beam_mask_v[0] = 0.
        _arange_b2_t = nnutils.get_arange_t(bsize*beam_size, 1)  # [*, 1]
        _arange_bs_t = nnutils.get_arange_t(bsize, 1)  # [bs, 1]
        # --
        # some states
        _shape_b2, _shape_b3 = [bsize, beam_size], [bsize, beam_size, beam_size]
        _prize_t = nnutils.input_real([_PRIZE] + [0.] * (beam_size - 1))  # [B]
        accu_score_t = nnutils.zeros(_shape_b2, dtype=torch.float32)  # not: use float32 to record this!!
        cur_step = 0
        # --
        # start looping
        # while True:
        for _ss in range(conf.test_max_step):
            # forward last slice
            hid_t, _ = bert.forward_model(cur_id_t[:, -1:], self_mask_k=cur_mask_t[:, -1:],
                                          cache=run_cache, cross_t=cross_t, cross_mask_k=cross_mask_k)  # [bs*B, 1, D]
            score_t = bert.forward_lmhead(hid_t).squeeze(-2)  # [bs*B, V]
            logprob_t = score_t.log_softmax(-1) * cur_mask_t[:, -1:]  # [*, V]
            if eos_penalty != 0.:  # directly change this on logprobs!
                logprob_t[..., eos_id] += eos_penalty
            # =====
            if conf.test_do_sample:  # simpler sampling
                # inner beam (each sample 1)
                _, inner_id = nnutils.category_sample(
                    logprob_t, dim=-1, keepdim=True, top_k=conf.test_sample_topk, top_p=conf.test_sample_topp)  # [*, 1]
                inner_id[cur_finished_t] = pad_id  # [*, 1], put pad
                inner_score = logprob_t.gather(-1, inner_id)  # [*, 1]
                inner_score[cur_finished_t] = 0.  # [*, 1]
                # outer: simply put them in
                re_score0 = inner_score.view(_shape_b2)  # [bs, B]
                re_id, re_score = inner_id.squeeze(-1), inner_score.squeeze(-1)  # [*]
            else:  # beam search
                # inner beam
                inner_score, inner_id = logprob_t.topk(beam_size, dim=-1, sorted=True)  # [*, B]
                inner_id[cur_finished_t] = pad_id  # [*, B], put pad
                inner_score[cur_finished_t] = _beam_mask_v  # [*, B]
                # beam selection
                _rr_b0, _rr_b1 = self._outer_beam(
                    accu_score_t, inner_score, cur_finished_t, cur_step, _NEG_INF, _shape_b3, _prize_t)  # [bs, B0], [bs, B1]
                # reindex at batch dimension
                if beam_size != 1:
                    batch_reidx_t = (_rr_b0 + _arange_bs_t * beam_size).view([-1])  # [*]
                    accu_score_t = accu_score_t[_arange_bs_t, _rr_b0]  # [bs, B]
                    cur_id_t = cur_id_t[batch_reidx_t]  # [bs*B, prev]
                    cur_mask_t = cur_mask_t[batch_reidx_t]  # [bs*B, prev]
                    cur_finished_t = cur_finished_t[batch_reidx_t]  # [bs*B]
                    # cache
                    for _ln, _lv in run_cache.items():  # layer
                        # todo(note): be careful about this, currently no need for cross!
                        if isinstance(_lv, dict):
                            _cv = _lv['self']
                            for _tn, _tv in _cv.items():  # specific tensor
                                _cv[_tn] = _tv[batch_reidx_t]
                    # --
                # gather the corresponding ones
                re_id0, re_score0 = inner_id.view(_shape_b3)[_arange_bs_t, _rr_b0, _rr_b1], \
                                    inner_score.view(_shape_b3)[_arange_bs_t, _rr_b0, _rr_b1]  # [bs, B], [bs, B]
                re_id, re_score = re_id0.view([-1]), re_score0.view([-1])  # [*], [*]
            # =====
            # check EOS & update states
            accu_score_t += re_score0
            cur_id_t = torch.cat([cur_id_t, re_id.unsqueeze(-1)], -1)  # [*, prev+1]
            cur_mask_t = torch.cat([cur_mask_t, (~cur_finished_t).to(nnutils.DEFAULT_FLOAT).unsqueeze(-1)], -1)  # [*, prev+1]
            cur_finished_t = (cur_finished_t | (re_id == eos_id) | (cur_mask_t.sum(-1) >= max_len_t))  # [*]
            cur_step += 1
            # finished?
            if cur_finished_t.all().item():
                break
        # --
        # final ranking
        final_score_t = accu_score_t.view([-1])
        if beam_size != 1:
            num_step_t = cur_mask_t.sum(-1).view(_shape_b2) - 1  # [bs, 1]
            _ranking_div = ((5+num_step_t)/(5+1)) ** conf.test_len_penalty  # [bs, 1]
            _ranking_score = ((accu_score_t + conf.test_len_reward * num_step_t) / _ranking_div)  # [bs, B]
            _, _rr_idx = _ranking_score.topk(beam_size, dim=-1, sorted=True)  # [bs, B]
            final_batch_reidx_t = (_rr_idx + _arange_bs_t * beam_size).view([-1])  # [*]
            # re-arrange
            final_id_t = cur_id_t[final_batch_reidx_t]
            final_mask_t = cur_mask_t[final_batch_reidx_t]
            final_score_t = final_score_t[final_batch_reidx_t]
        else:
            final_id_t, final_mask_t = cur_id_t, cur_mask_t
        # --
        # assign the results
        arr_id, arr_mask, arr_score = [nnutils.get_value(z) for z in [final_id_t, final_mask_t, final_score_t]]
        _inst_setter = self.ztask._inst_setter
        _all_stage = 0
        for bidx, inst in enumerate(med.ibatch.insts):
            _ii = bidx * beam_size  # already sorted
            best_id, best_mask = arr_id[_ii], arr_mask[_ii] > 0.
            list_id = best_id[best_mask].tolist()
            list_token = self.tokenizer.convert_ids_to_tokens(list_id)
            _inst_setter(inst, list_token, list_id)
            inst.info.update({'num_stage': len(list_id)-1})
            _all_stage += inst.info['num_stage']
            # --
            if conf.test_output_all:  # further export all ones!
                res = []
                for _jj in range(beam_size):
                    _this_id, _this_mask = arr_id[_ii+_jj], arr_mask[_ii+_jj] > 0.
                    _this_toks = self.tokenizer.convert_ids_to_tokens(_this_id[_this_mask].tolist())
                    _this_score = arr_score[_ii+_jj].item()
                    res.append({"tok": _this_toks, "score": _this_score, "stages": list(range(len(_this_toks)))})
                if 'res' not in inst.info:
                    inst.info['res'] = []
                inst.info['res'].extend(res)
            # --
        # --
        return {"all_stage": _all_stage}

    def _outer_beam(self, accu_score_t, inner_score, cur_finished_t, cur_step: int, _NEG_INF, _shape_b3, _prize_t):
        beam_size = _shape_b3[-1]
        # prepare extra score for ranking
        _extra_ranking_score = nnutils.zeros(_shape_b3)  # [bs, B0, B1] temp scored added for special treatment!
        if cur_step == 0:  # mask out non0s at the first step!
            _extra_ranking_score[:, 1:] = _NEG_INF
        one_is_finished_b = cur_finished_t.view(_shape_b3[:2])  # [bs, B0]
        _extra_ranking_score[one_is_finished_b] += _prize_t  # keep finished ones as they are!
        # rank them
        _ranking_score0 = accu_score_t.unsqueeze(-1) + inner_score.view(_shape_b3) + _extra_ranking_score  # [bs, B0, B1]
        _ranking_score = _ranking_score0.view(_shape_b3[:-2] + [-1])  # [bs, B0xB1]
        _, _rr_idx = _ranking_score.topk(beam_size, dim=-1, sorted=True)  # [bs, B]
        _rr_b0, _rr_b1 = _rr_idx // beam_size, _rr_idx % beam_size  # [bs, B0], [bs, B1]
        return _rr_b0, _rr_b1  # [bs, B0], [bs, B1]

# --
# b zgen/model/tasks/dec_slm:70
