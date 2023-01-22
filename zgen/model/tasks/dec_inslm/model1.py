#

# model1 (based on abs-embedding, but need to forward all for every stage)

__all__ = [
    "ZDecoderInslm1Conf", "ZDecoderInslm1",
]

from typing import List
import torch
import torch.nn.functional as F
import numpy as np
from zgen.utils import nn as nnutils
from .base import ZDecoderInslmConf, ZDecoderInslm
from ...core import *

# --
class ZDecoderInslm1Conf(ZDecoderInslmConf):
    def __init__(self):
        super().__init__()
        # --
        # some specific one only for model1
        self.fd_noi_first = True  # output NOI at first possibility
        self.fd_blank_topk = 0  # output tok by topk, <=0 means all!
        # --

@node_reg(ZDecoderInslm1Conf)
class ZDecoderInslm1(ZDecoderInslm):
    def __init__(self, conf: ZDecoderInslm1Conf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZDecoderInslm1Conf = self.conf
        # --

    def _calc_output(self, idxes_t, masks_t):
        # pad an extra one: will not influence results since masked out
        _pad_pat = [0,1]
        pad_idxes_t = F.pad(idxes_t, _pad_pat, value=self.pad_token_id)  # [*, klen+1]
        pad_masks_t = F.pad(masks_t, _pad_pat, value=0.)  # [*, klen+1]
        # --
        hid_t = self.forward_bert_model(input_ids=pad_idxes_t, self_mask_k=pad_masks_t)[0]  # [*, klen+1, D]
        feat_t = self.comb([hid_t[..., :-1, :], hid_t[..., 1:, :]])  # [*, klen, D]
        out_t = self.bert.lmhead(feat_t)  # [*, klen, V]
        return out_t

    # --
    # for the masks
    """
    For example: 
    idxes_t: [CLS] x1 [x2] x3 x4 [x5] [SEP] [PAD]
    kept_t: 1 0 1 0 0 1 1 0
    kept_idxes/masks: 0 2 5 6 ;; 1 1 1 1 -> input to model
    # _expanded_idxes: 0 2 5 6 7, _seg_sizes: 2 3 1 1
    seg_idxes: [0,1] [2,3,4] [5] [6]
    """
    def do_loss(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderInslmConf = self.conf
        # --
        # first prepare input!
        IDX_PAD = self.pad_token_id
        ids = [inst.idxes for inst in med.ibatch.insts]
        idxes_t, masks_t = nnutils.go_batch_2d(ids, IDX_PAD)  # [*, elen]
        _shape = list(idxes_t.shape)  # [*, elen]
        # --
        # prepare rollin and oracle scores
        rollin_scores = self.rollin_helper.get_oracle(idxes_t, masks_t)  # [*, elen]
        oracle_scores = self.oracle_helper.get_oracle(idxes_t, masks_t) \
            if conf.oracle_strategy!=conf.rollin_strategy else rollin_scores  # [*, elen]
        # sample and obtain segments
        kept_idxes_t, kept_masks_t, seg_idxes, seg_mask_t = self._sample_and_seg(idxes_t, masks_t, rollin_scores)  # [*, klen, (S)]
        # forward model
        kept_ids_t = idxes_t.gather(-1, kept_idxes_t)  # [*, klen]
        kept_ids_t[kept_masks_t<=0.] = IDX_PAD
        out_t = self._calc_output(kept_ids_t, kept_masks_t)  # [*, klen, V]
        nll_t = - out_t.log_softmax(dim=-1)  # [*, klen, V]
        # gather loss
        final_loss_t = self._gather_loss(nll_t, idxes_t, seg_idxes, seg_mask_t, oracle_scores)  # [*, klen]
        final_mask_t = kept_masks_t * (kept_ids_t != self.sep_token_id).to(nnutils.DEFAULT_FLOAT)  # no after EOS!
        _loss_item = LossHelper.compile_leaf_loss(
            f'{self.name}_nll', (final_loss_t*final_mask_t).sum(), final_mask_t.sum(), loss_lambda=conf.loss_inslm)
        # --
        ret = LossHelper.combine_multiple_losses([_loss_item])
        return ret, {}

    def _sample_and_seg(self, idxes_t, masks_t, rollin_scores_t):
        _shape = list(idxes_t.shape)  # [*, elen]
        # roll-in: first sample mask rate
        _mrate_low, _mrate_high = self.mrate_low.value, self.mrate_high.value
        cur_mrates = _mrate_low + (_mrate_high-_mrate_low) * nnutils.rand(_shape[:-1]).unsqueeze(-1)  # [*, 1]
        # then sample for the tokens and segment
        # -- originally only random rollin
        # kept_mask_b = ((masks_t>0) & ((nnutils.rand(_shape) >= cur_mrates) | (idxes_t == self.cls_token_id)
        #                               | (idxes_t == self.sep_token_id)))  # [*, elen]
        # --
        # now use rollin_scores
        # slightly allow the topk to be slightly larger, use 'sum*rates' rather than '(sum-2)*rates'
        sel_topk_t = (masks_t.sum(-1, keepdims=True) * (1.-cur_mrates)).to(nnutils.DEFAULT_INT)  # [*, 1]
        _is_special = (idxes_t == self.cls_token_id) | (idxes_t == self.sep_token_id)  # special ones!
        _mask_t2 = masks_t * (1.-_is_special.to(nnutils.DEFAULT_FLOAT))  # exclude the spcial ones
        sel_mask_t = nnutils.select_topk(rollin_scores_t, sel_topk_t, _mask_t2, noise=self.rollin_noise)  # [*, elen]
        sel_mask_b = ((masks_t>0) & ((sel_mask_t>0) | _is_special))
        # --
        len_t = masks_t.long().sum(-1, keepdims=True)  # [*, 1]
        kept_idxes_t, _, kept_masks_t, seg_idxes, seg_mask_t = self._valid2seg(sel_mask_b, len_t)
        return kept_idxes_t, kept_masks_t, seg_idxes, seg_mask_t  # 2x[*, klen], 2x[*, klen, S]

    def _gather_loss(self, nll_t, orig_ids_t, seg_idxes, seg_mask_t, oracle_scores_t):
        # first gather full loss
        _seg_shape = list(seg_idxes.shape)  # [*, klen, S]
        trg_ids_t = orig_ids_t.gather(-1, seg_idxes.view(_seg_shape[:-2]+[-1])).view(_seg_shape)  # [*, klen, S]
        trg_ids_t[..., 0] = self.noi_token_id  # put NOI at first
        full_loss0_t = nll_t.gather(-1, trg_ids_t)  # [*, klen, S]
        # next weight loss
        is_noi = (seg_mask_t.sum(-1) == 1).to(nnutils.DEFAULT_FLOAT)  # [*, klen], only the leading one means NOI!
        trg_mask_t = seg_mask_t.clone()  # [*, slen, S]
        trg_mask_t[..., 0] = is_noi  # if no others, then NOI!!
        # collect loss
        final_loss_t = self._collect_loss(full_loss0_t, trg_mask_t, self.oracle_specs, oracle_scores_t, seg_idxes=seg_idxes)
        return final_loss_t   # [*, slen]

    # --
    """
    For example: 
    idxes_t: [CLS] x1 x2 x3 x4 x5 [SEP]
    stages:     0 2 3 1 3 2 0
    tok_scores: 0. ... ...  0.
    noi_scores: ... ... ... 0.
    """
    def do_predict(self, med: ZMediator, *args, **kwargs):
        # todo(note): for eval, no requirements for speed, thus simply use for-loop
        conf: ZDecoderInslmConf = self.conf
        _noi_greedy = conf.fd_noi_first
        # --
        ID_PAD, ID_NOI = self.pad_token_id, self.noi_token_id
        # first prepare inputs
        ids = [inst.idxes for inst in med.ibatch.insts]
        idxes_t, masks_t = nnutils.go_batch_2d(ids, ID_PAD)  # [*, elen]
        impt_t = self.fd_helper.get_oracle(idxes_t, masks_t)  # static-oracle [*, elen] or None
        if impt_t is not None:
            impt_arr = nnutils.get_value(impt_t)
        else:
            impt_arr = None
        # _shape = list(idxes_t.shape)  # [*, elen]
        inst_states = []
        for inst in med.ibatch.insts:
            _len_m2 = len(inst) - 2
            assert _len_m2 >= 0
            # (stages, tok-scores, noi-scores)
            inst_states.append(([0]+[None]*_len_m2+[0], [0.]+[None]*_len_m2+[0.], [None]+[None]*_len_m2+[0.]))
        # start the forced decoding
        cur_stage = 1
        while True:  # after fill out all NOIs
            # forward model
            cur_ids = [[z for z,s in zip(inst.idxes, states[0]) if s is not None]
                       for inst, states in zip(med.ibatch.insts, inst_states)]  # put the non-None ones!
            cur_idxes_t, cur_masks_t = nnutils.go_batch_2d(cur_ids, ID_PAD)  # [*, cur_len]
            cur_out_t = self._calc_output(cur_idxes_t, cur_masks_t)  # [*, cur_len, V]
            cur_logprob_t = cur_out_t.log_softmax(-1)  # [*, cur_len, V]
            # use forloop to pick them!
            arr_logprob = nnutils.get_value(cur_logprob_t)  # [*, cur_len, V]
            cur_finished = True
            for bidx, inst in enumerate(med.ibatch.insts):
                one_stages, one_tok_scores, one_noi_scores = inst_states[bidx]
                if all(z is not None for z in one_noi_scores):
                    continue  # already finished all (tok and noi), no need to check!
                one_tok_finished = all(z is not None for z in one_stages)  # already finished all toks, remaining NOI
                cur_finished = cur_finished and one_tok_finished  # if all toks finished at input, no need for next round
                one_logprobs = arr_logprob[bidx]  # [cur_len, V]
                # check each blank
                one_blank_idx = 0
                one_blanks = [[]]  # for each blank [list((widx, logprob)), ...]
                for one_tidx in range(1, len(one_stages)):
                    this_logprobs = one_logprobs[one_blank_idx]  # [V]
                    if one_stages[one_tidx] is not None:  # input valid
                        # NOI
                        if one_stages[one_tidx-1] is not None and one_noi_scores[one_tidx-1] is None \
                                and (_noi_greedy or one_tok_finished):  # can put NOI here!
                            one_noi_scores[one_tidx-1] = this_logprobs[ID_NOI].item()
                        # a new blank!
                        one_blank_idx += 1
                        one_blanks.append([])
                    else:  # inside blank
                        one_blanks[-1].append((one_tidx, this_logprobs[inst.idxes[one_tidx]].item()))
                # select and put tokens!
                selected_pairs = self._select_blanks(inst, one_blanks, (impt_arr[bidx] if impt_arr is not None else None))
                for _tidx, _logprob in selected_pairs:
                    assert one_stages[_tidx] is None and one_tok_scores[_tidx] is None
                    one_stages[_tidx] = cur_stage
                    one_tok_scores[_tidx] = _logprob
            # --
            if cur_finished:
                break
            cur_stage += 1
        # --
        # put results
        for _inst, _states in zip(med.ibatch.insts, inst_states):
            _inst.info['inslm_info'] = _states
        return {}

    def _select_blanks(self, inst, blanks: List, impt_arr):
        conf: ZDecoderInslmConf = self.conf
        _fd_blank_topk = conf.fd_blank_topk
        # --
        all_cands = []
        for one_pairs in blanks:  # select local top1, note: can only select one at one blank
            cur_cands = []
            _cent, _ii = (len(one_pairs)-1)//2, 0
            for _tidx, _logprob in one_pairs:
                stra_model, stra_bt, stra_alpha = self.fd_specs
                if stra_model:
                    _score = _logprob * stra_alpha
                elif stra_bt:
                    _score = - abs(_cent - _ii) * stra_alpha
                else:  # no need to rev here!
                    _score = impt_arr[_tidx]
                cur_cands.append((_score, _logprob, _tidx))  # rank by score
                _ii += 1
            if len(cur_cands) > 0:
                _max_one = max(cur_cands)
                all_cands.append(_max_one)
        # select global topk
        if _fd_blank_topk <= 0 or _fd_blank_topk >= len(all_cands):
            _sels = all_cands  # all are ok!
        else:  # topk
            _aggr_scores = np.asarray([-z[0] for z in all_cands])  # if topk, then neg to make it select smaller ones
            _par_idxes = np.argpartition(_aggr_scores, _fd_blank_topk)
            _sels = [all_cands[ii] for ii in _par_idxes[:_fd_blank_topk]]
        # return
        return [(z[2], z[1]) for z in _sels]
        # --

# --
# b zgen/model/tasks/dec_inslm/model1:?
