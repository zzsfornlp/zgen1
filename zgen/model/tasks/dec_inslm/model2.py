#

# model2 (based on special tree-ins-embeddings)

__all__ = [
    "ZDecoderInslm2Conf", "ZDecoderInslm2",
]

import torch
from zgen.utils import nn as nnutils
from .base import ZDecoderInslmConf, ZDecoderInslm
from ...core import *
from .helper_incr import IncrCacheFd, IncrEmbedConf

# --
# note: actually no seq in this mode!
class ZDecoderInslm2Conf(ZDecoderInslmConf):
    def __init__(self):
        super().__init__()
        # --
        self.incr_embed = IncrEmbedConf()
        # --

@node_reg(ZDecoderInslm2Conf)
class ZDecoderInslm2(ZDecoderInslm):
    def __init__(self, conf: ZDecoderInslm2Conf, ztask, zmodel, **kwargs):
        if conf.bconf is not None:
            conf.bconf.no_posi_embeds = True  # no ordinary position embeddings!
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZDecoderInslm2Conf = self.conf
        # --
        self.incr_embed = conf.incr_embed.make_node(esize=self.bert.hidden_size)
        # --

    def do_loss(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderInslm2Conf = self.conf
        # prepare input
        IDX_PAD = self.pad_token_id
        ids = [inst.idxes for inst in med.ibatch.insts]
        ids_t, masks_t = nnutils.go_batch_2d(ids, IDX_PAD)  # [*, elen]
        # prepare rollin and oracle scores
        rollin_scores = self.rollin_helper.get_oracle(ids_t, masks_t)  # [*, elen]
        oracle_scores = self.oracle_helper.get_oracle(ids_t, masks_t) \
            if conf.oracle_strategy!=conf.rollin_strategy else rollin_scores  # [*, elen]
        # get loss by forced decoding
        final_loss_t, final_mask_t = \
            self._forced_decoding(ids_t, masks_t, self.rollin_specs, rollin_scores, self.oracle_specs, oracle_scores)[:2]
        _loss_item = LossHelper.compile_leaf_loss(
            f'{self.name}_nll', (final_loss_t*final_mask_t).sum(), final_mask_t.sum(), loss_lambda=conf.loss_inslm)
        # --
        ret = LossHelper.combine_multiple_losses([_loss_item])
        return ret, {}

    def do_predict(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderInslm2Conf = self.conf
        # prepare input
        IDX_PAD = self.pad_token_id
        ids = [inst.idxes for inst in med.ibatch.insts]
        ids_t, masks_t = nnutils.go_batch_2d(ids, IDX_PAD)  # [*, elen]
        # prepare rollin and oracle scores
        rollin_scores = self.fd_helper.get_oracle(ids_t, masks_t)  # [*, elen]
        # do forced decoding (rollin == oracle)
        res = self._forced_decoding(ids_t, masks_t, self.fd_specs, rollin_scores, None, None)  # [*, AK]
        # put results!
        arr_nll, arr_mask, arr_left, arr_right, arr_idxes, arr_stages = [nnutils.get_value(z) for z in res]
        for bidx, inst in enumerate(med.ibatch.insts):
            _len_m2 = len(inst) - 2
            _stage = arr_stages[bidx].tolist()[:_len_m2+2]
            _tok_scores, _noi_scores = [0.]+[None]*_len_m2+[0.], [None]+[None]*_len_m2+[0.]
            for ii, mm in enumerate(arr_mask[bidx]):
                if mm == 0: continue  # ignore nonvalid ones
                vv, ileft, iright, isel = [z[bidx, ii].item() for z in [arr_nll, arr_left, arr_right, arr_idxes]]
                if isel == 0:
                    assert ileft+1 == iright and _noi_scores[ileft] is None
                    _noi_scores[ileft] = -vv  # negate it(nll) for log-prob
                else:
                    assert _tok_scores[isel] is None
                    _tok_scores[isel] = -vv
            inst.info['inslm_info'] = (_stage, _tok_scores, _noi_scores)
        return
        # --

    # [*, elen]
    def _forced_decoding(self, ids_t, masks_t, rollin_specs, rollin_scores, oracle_specs, oracle_scores):
        # more preparations of input
        special_ids_t = ids_t.clone()  # [*, elen]
        special_ids_t[..., 0] = self.noi_token_id  # note: put NOI at idx0 for convenience!
        len_t = masks_t.to(nnutils.DEFAULT_INT).sum(-1, keepdims=True)  # [*, 1]
        _len_t_m1 = len_t - 1
        _shape = list(ids_t.shape)  # [*, elen]
        # overall status
        cur_stages_t = nnutils.constants(_shape, -1, dtype=nnutils.DEFAULT_INT)  # [*, elen]
        cur_stages_t[..., 0] = 0  # assume stage0 for thes two!
        cur_stages_t.scatter_(-1, _len_t_m1, 0)
        # rolling!
        _imap_t = nnutils.constants(_shape, -1, dtype=nnutils.DEFAULT_INT)  # [*, elen], orig_idx -> running_idx
        _imap_t[..., 0] = 0  # idx0=CLS
        _imap_t.scatter_(-1, _len_t_m1, 1)  # idx1=SEP
        cache = IncrCacheFd(self, ids_t, _imap_t, _len_t_m1)   # running cache
        _NEG_INF = float('-inf')
        cur_stage = 1
        rstra_model, rstra_bt, rstra_alpha = rollin_specs
        _arange_t = nnutils.arange(ids_t.shape[-1], dtype=nnutils.DEFAULT_INT)  # [elen]
        all_pieces = []
        cur_closed_t = self._stage2close(cur_stage, cur_stages_t, len_t)  # [*, elen]
        while True:
            # obtain blanks: [*, k]
            one_left_t, one_right_t, one_vmasks_t, _, _ = self._valid2seg(cur_stages_t>=0, len_t, cur_closed_t, ret_seg=False)
            all_pieces.append((one_left_t, one_right_t, one_vmasks_t))
            one_is_noi = ((one_left_t+1) >= one_right_t).to(nnutils.DEFAULT_FLOAT)  # [*, k], no space
            # obtain non-noi blanks: [*, kk]
            sel_masks_t0 = one_vmasks_t * (1. - one_is_noi)  # [*, k], mask excluding NOI
            sel_masks_t, sel_left_t, sel_right_t = \
                self._select_and_compress(sel_masks_t0, one_left_t, one_right_t, pad=0)  # [*, kk]
            sel_cand_masks = ((sel_left_t.unsqueeze(-1) < _arange_t) & (_arange_t < sel_right_t.unsqueeze(-1)))\
                .to(nnutils.DEFAULT_FLOAT)  # [*, kk, elen], valid candidates
            # obtain scores for selecting: [*, kk, elen]
            if rstra_model:  # need to forward the model!
                cache.refresh_forward()
                with nnutils.no_grad_env():
                    _sel_score_tV = cache.score(sel_left_t, sel_right_t)  # [*, k, V]
                    _sel_score_t = self._select_scores(_sel_score_tV, special_ids_t)  # [*, k, elen]
                _sel_score_t = _sel_score_t * rstra_alpha
            elif rstra_bt:
                _center = ((_arange_t * sel_cand_masks).sum(-1) // sel_cand_masks.sum(-1).clamp(min=1)).unsqueeze(-1)  # [*, k, 1]
                _sel_score_t = - (_center - _arange_t).abs()  # [*, k, elen], distance to center
                _sel_score_t = _sel_score_t * rstra_alpha
            else:
                _sel_score_t = rollin_scores.unsqueeze(-2)  # [*, 1, elen]
            # mask out invalid ones
            _extra_ones = torch.zeros_like(sel_cand_masks)
            _extra_ones[sel_cand_masks <= 0.] = _NEG_INF
            _sel_score_t = _sel_score_t + _extra_ones
            # select it; todo(+N): currently simply take maximum!
            _, sel_idxes_t = _sel_score_t.max(-1)  # [*, kk]
            sel_idxes_t[sel_masks_t<=0.] = 0  # make sure not strange idxes!
            # update
            # overall status
            cur_stages_t.scatter_(-1, sel_idxes_t, cur_stage)
            cur_stages_t[..., 0] = 0  # reset for pad(0)
            # calculate closed_t on the fly!
            cur_closed_t = self._stage2close(cur_stage, cur_stages_t, len_t)
            if (cur_closed_t>0).all().item():  # finished when all closed
                break  # if ok, no need to further feed!
            # incr. status
            cache.append(sel_idxes_t, sel_masks_t, sel_left_t, sel_right_t)  # imap is updated inside!
            # --
            cur_stage += 1
        # --
        # score them!
        cache.refresh_forward()
        # concat and then compress again to save memory: [*, AK0] -> [*, AK]
        all_left_t0, all_right_t0, all_vmasks_t0 = [torch.cat([z[i] for z in all_pieces], -1) for i in range(3)]  # [*, AK0]
        all_vmasks_t, all_left_t, all_right_t = self._select_and_compress(all_vmasks_t0, all_left_t0, all_right_t0, pad=0)
        all_score_tV = cache.score(all_left_t, all_right_t)  # [*, AK, V]
        all_score_nll = - all_score_tV.log_softmax(dim=-1)  # [*, AK, V]
        all_cand_nll = self._select_scores(all_score_nll, special_ids_t)  # [*, AK, elen]
        # get cands
        all_cand_masks = ((all_left_t.unsqueeze(-1) < _arange_t) & (_arange_t < all_right_t.unsqueeze(-1)))\
            .to(nnutils.DEFAULT_FLOAT)  # [*, AK, elen], valid candidates
        all_is_noi = (all_cand_masks.sum(-1) <= 0.).to(nnutils.DEFAULT_FLOAT) * all_vmasks_t  # [*, AK]
        all_cand_masks[..., 0] = all_is_noi  # note: idx0 is special one: NOI!
        # gather loss!
        if oracle_specs is None:  # gather rollin ones (used when testing)
            # simply take the ones with the lowest stage!
            _rollin_stages = ((1-all_cand_masks) * (_shape[-1]+100) + cur_stages_t.unsqueeze(-2))  # [*, AK, elen]
            _, _rollin_idxes = _rollin_stages.min(-1, keepdims=True)  # [*, AK, 1]
            trg_weight_t = torch.zeros_like(all_cand_nll)
            trg_weight_t.scatter_(-1, _rollin_idxes, 1.)  # [*, AK, elen]
            trg_weight_t *= all_cand_masks
            loss_t = (all_cand_nll * trg_weight_t).sum(-1)  # [*, AK]
            rollin_idxes = _rollin_idxes.squeeze(-1)  # [*, AK]
        else:
            loss_t = self._collect_loss(all_cand_nll, all_cand_masks, oracle_specs, oracle_scores, seg_idxes=None)
            rollin_idxes = None  # no need in this mode
        # --
        return loss_t, all_vmasks_t, all_left_t, all_right_t, rollin_idxes, cur_stages_t  # *[*, AK], [*, elen]

    # specially prepare scores: [*, ??, V], [*, L]
    def _select_scores(self, scores_t, ids_t):
        _shape0 = list(scores_t.shape)
        _shape1 = list(ids_t.shape)
        for _ in range(len(_shape0) - len(_shape1)):
            ids_t = ids_t.unsqueeze(-2)  # [*, *1, L]
        _sel_t = ids_t.expand(_shape0[:-1] + [_shape1[-1]])  # [*, ??, L]
        ret = scores_t.gather(-1, _sel_t)  # [*, ??, L]
        return ret

# --
# b zgen/model/tasks/dec_inslm/model2:?
