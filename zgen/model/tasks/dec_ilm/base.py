#

# ilm

__all__ = [
     "ZTaskIlmConf", "ZTaskIlm", "ZDecoderIlmConf", "ZDecoderIlm",
]

from collections import OrderedDict
import os
import torch
from zgen.utils import zlog, zwarn, ScheduledValue, SVConf, MyStat, zglob1, Constants, Random
from zgen.utils import nn as nnutils
from ...core import *
from ..base import *
from .help_incr import *
from .help_oracle import *
from .help_search import Searcher
from .help_others import *

# --

class ZTaskIlmConf(ZTaskBaseTConf):
    def __init__(self):
        super().__init__()
        self.name = "ilm"
        self.ilm_conf = ZDecoderIlmConf()
        # --

    def build_task(self):
        return ZTaskIlm(self)

class ZTaskIlm(ZTaskBaseT):
    def __init__(self, conf: ZTaskIlmConf):
        super().__init__(conf)
        # --

    def build_mod(self, model):
        return self.conf.ilm_conf.make_node(self, model)
# --

class ZDecoderIlmConf(ZModBaseTConf):
    def __init__(self):
        super().__init__()
        # --
        self.loss_ilm = 1.
        self.conf_incr = IncrModelConf()
        # --
        # noi weight
        self.loss_noi_weight = 1.  # down-weight NOI?
        self.close_only_all_noi = False  # close the seq at once only when all slots predict NOI?
        self.loss_unlike = 0.  # unlikelihood loss
        self.ps_conf = PsHelperConf()  # make it general for post-processing of the oracles!
        self.ho_conf = HelperOracleConf()  # some confs for specific strategies!
        # --
        # special distill with extra_model
        self.soft_target_with_slm = False
        # --
        # training roll-in and oracle
        self.rollin_strategy = "random0"  # l2r/freq=(log(1+f))/tfidf/random/bt + R?
        self.rollin_tau = SVConf.direct_conf(val=1.)  # 0 means simply maximum!
        self.oracle_strategy = "uniform0"  # l2r/freq/tfidf/uniform/bt/model + R?
        self.oracle_tau = SVConf.direct_conf(val=1.)
        self.oracle_mixu = SVConf.direct_conf(val=0.)  # mix uniform?
        self.oracle_take_max = False  # note: deprecated by "setting oracle_tau=0."
        self.oracle_model_detach = True  # detach for p?
        self.stat_file = ""
        self.span_size_alpha = 0.  # special alpha for score (both rollin & oracle): score/(span_size)**alpha
        # other training options
        self.train_compress_thresh = 0.5  # compress thresh for cache (when <this)
        self.train_num_stages = Constants.INT_PRAC_MAX  # how many stages for training? (sample continuous ones)
        self.train_detach_growing = True  # save memory for backward by detaching before from_stage
        self.train_reweight_stage = False  # reweighting for each stage
        self.oracle_model_detach = True  # detach for oracle_model
        self.train_onestep_model0 = False  # specifically for model0, random rollin & one-step training
        self.train_rollin_noise = 0.  # rollin noise to make some variations?
        self.train_smk_low = 1.  # sample-mask-keep low thresh
        self.train_curriculum = SVConf.direct_conf(val=1., min_val=0., max_val=1.)  # from 0 to 1, by default always 1
        # test
        self.test_compress_thresh = 0.5
        self.test_beam_size = 4
        self.test_max_len = 500
        self.test_max_step = 100000
        self.test_max_ratio = 1.5  # max ratio to src if there are
        self.test_noi_penalty = 0.  # depends on specific tasks!
        self.test_mid_div = 'nope'  # nope/tok/ins
        self.test_final_div = 'nope'  # nope/tok/ins
        self.test_record_history = False  # maybe slightly faster
        self.test_dtp = -1.  # delay threshold prob
        # test with sample
        self.test_do_sample = False
        self.test_sample_topk = 0
        self.test_sample_topp = 0.
        self.test_sample_max_bs = 128  # sample how many at each time?
        # --
        # special debug option
        self.debug_ilm = False
        # test output
        self.test_output_all = False
        # --

@node_reg(ZDecoderIlmConf)
class ZDecoderIlm(ZModBaseT):
    def __init__(self, conf: ZDecoderIlmConf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZDecoderIlmConf = self.conf
        # --
        self.incr = conf.conf_incr.make_node(mod=self)  # simply pass self in
        if conf.train_onestep_model0:
            assert self.incr.is_model0, "Must be model0 to use this mode!!"
            assert conf.rollin_strategy.startswith("random"), "Currently only support random rollin in this mode!"
        # --
        self.rollin_tau = ScheduledValue(f"rollin_tau", conf.rollin_tau)
        if conf.oracle_take_max:
            zwarn(f"Deprecated option oracle_take_max, make oracle_tau {conf.oracle_tau.val} -> 0. instead")
            conf.oracle_tau.val = 0.
        self.oracle_tau = ScheduledValue(f"oracle_tau", conf.oracle_tau)
        self.oracle_mixu = ScheduledValue(f"oracle_mixu", conf.oracle_mixu)
        self.rollin_specs = self._check_special_strategies(conf.rollin_strategy)
        self.oracle_specs = self._check_special_strategies(conf.oracle_strategy)
        self.ps_helper = PsHelper(conf.ps_conf)
        self.train_curriculum = ScheduledValue(f"train_curriculum", conf.train_curriculum)
        # --
        # oracle helpers
        self.stat = None
        if conf.stat_file:
            sfile = zglob1(conf.stat_file, check_iter=10)
            if os.path.isfile(sfile):
                self.stat = MyStat.create_from_file(sfile)
            else:
                zwarn(f"No such file: {sfile}")
        _vocab = self.tokenizer.get_vocab()
        self.rollin_helper = HelperOracle(conf.rollin_strategy, self.stat, _vocab, conf.ho_conf)
        if conf.oracle_strategy.startswith("random"):
            zlog(f"Change oracle_strategy from {conf.oracle_strategy} to uniform0!!")
            conf.oracle_strategy = "uniform0"  # just use uniform as oracle rather than random!!
        self.oracle_helper = HelperOracle(conf.oracle_strategy, self.stat, _vocab, conf.ho_conf)
        # --
        # searcher
        self.searcher = Searcher(self)
        _gen = Random.get_generator('train')
        self.random_streamer = Random.stream(_gen.random)
        # --

    def set_test_noi_penalty(self, p: float):
        old_p = self.conf.test_noi_penalty
        self.conf.test_noi_penalty = p
        zlog(f"[!!] Set NOI-penalty from {old_p} to {self.conf.test_noi_penalty}")
        return self.conf.test_noi_penalty
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

    def set_test_confs(self, sample_p: float = None, noi_p: float = None, *args, **kwargs):
        if sample_p is not None:
            self.set_test_sample_p(sample_p)
        if noi_p is not None:
            self.set_test_noi_penalty(noi_p)
        # --

    def _get_scheduled_values(self):
        return OrderedDict([("_rollin_tau", self.rollin_tau),
                            ("_oracle_tau", self.oracle_tau), ("_oracle_mixu", self.oracle_mixu),
                            ("_center_mixu", self.ps_helper.center_mixu),
                            ("_train_curriculum", self.train_curriculum),])

    def _check_special_strategies(self, name: str):
        stra_alpha = -1 if (name[-1] == "1") else 1
        stra_name = name[:-1]
        stra_model, stra_bt, stra_ps = [stra_name==z for z in ["model", "bt", "ps"]]
        return stra_model, stra_bt, stra_ps, stra_alpha

    # *[*, L] -> [*, L, V]
    def _get_soft_target_with_slm(self, med, ids_t, masks_t):
        from zgen.drive import extra_models
        slm_model = extra_models[0]  # note: use the first extra model!
        tmp_med = ZMediator()
        tmp_med.restart(med.ibatch)  # borrow ibatch
        slm_model.Menc.do_prep(tmp_med)  # enc-prep
        slm_model.Menc.do_loss(tmp_med)  # enc-run
        prob_t = slm_model.Mslm.calc_output(tmp_med, ids_t, masks_t).softmax(-1)  # [*, L, V]
        _slice_shape = list(prob_t.shape)
        _slice_shape[-2] = 1  # [*, 1, V]
        prob_t2 = torch.cat([nnutils.zeros(_slice_shape), prob_t[..., :-1, :]], -2)  # [*, 1+L-1, V]
        prob_t2[..., 0, self.ztask.special_vocab.noi_token_id] = 1.  # note: idx=0 as NOI!!
        return prob_t2

    def do_loss(self, med: ZMediator, *args, **kwargs):
        # todo(note): [DEBUG] turn off dropout for debug
        # [NOPE] self.eval()
        # --
        conf: ZDecoderIlmConf = self.conf
        _debug = conf.debug_ilm
        if _debug:
            self.eval()  # no dropout when debugging
        # prepare input
        ID_PAD = self.tokenizer.pad_token_id
        ids = self.get_inst_idxes(med.ibatch.insts)
        ids_t, masks_t = nnutils.go_batch_2d(ids, ID_PAD)  # [*, L]
        # --
        # soft target?
        soft_trg_t = None
        if conf.soft_target_with_slm:
            with nnutils.no_grad_env():
                soft_trg_t = self._get_soft_target_with_slm(med, ids_t, masks_t)  # [*, L, V]
        # --
        # prepare cross feats
        if conf.cross_mname:
            cross_t, cross_mask_k = med.get_cache((conf.cross_mname, 'enc_mem')), med.get_cache((conf.cross_mname, 'enc_mask'))
        else:
            cross_t, cross_mask_k = None, None
        # --
        if _debug: zlog("Before ilm.run")
        if conf.train_onestep_model0:  # random rollin
            fd_canvas = self.searcher.step_random_decoding(ids_t, masks_t, self.train_curriculum.value)
            run_feats = self.incr.run_canvas(fd_canvas, 1, 1, conf.train_detach_growing,
                                             cross_t=cross_t, cross_mask_k=cross_mask_k)  # *[*, ??, D]
        else:
            # prepare rollin scores
            rollin_scores = self.rollin_helper.get_oracle(ids_t, masks_t, med)  # [*, L]
            # do forced decoding
            if _debug: zlog("Before forced_decoding.")
            fd_canvas = self.searcher.forced_decoding(
                ids_t, masks_t, self.rollin_specs, rollin_scores, conf.train_rollin_noise, self.rollin_tau.value,
                compress_thresh=conf.train_compress_thresh, close_only_all_noi=conf.close_only_all_noi, smk_low=conf.train_smk_low,
                cross_t=cross_t, cross_mask_k=cross_mask_k, debug_do_forw=_debug)
            # forward in batch and collect loss
            # sample stages and run feats
            _train_num_stage = conf.train_num_stages
            _cur_num_stage = fd_canvas.cur_stage  # range is [1, N]
            if _train_num_stage >= _cur_num_stage:
                from_stage, to_stage = 1, _cur_num_stage
            else:  # sample a piece
                rr = next(self.random_streamer)  # sample a number [0, 1)
                ss = _cur_num_stage - _train_num_stage + 1  # number of selections
                from_stage = int(rr*ss) + 1
                to_stage = from_stage + _train_num_stage - 1
            if _debug: zlog("Before run_canvas.")
            _valid_feats = self.incr.run_canvas(fd_canvas, from_stage, to_stage, conf.train_detach_growing,
                                                cross_t=cross_t, cross_mask_k=cross_mask_k)  # *[*, ??, D]
            run_feats = [None] * (from_stage-1) + _valid_feats  # simply padding previous ones
        # gather loss
        oracle_scores = self.oracle_helper.get_oracle(ids_t, masks_t, med) \
            if conf.oracle_strategy!=conf.rollin_strategy else rollin_scores  # [*, L]
        if _debug: zlog("Before collect_loss.")
        loss_items = self._collect_loss(fd_canvas, run_feats, self.oracle_specs, oracle_scores, soft_trg_t)
        # --
        ret = LossHelper.combine_multiple_losses(loss_items)
        if _debug: zlog("Before return.")
        return ret, {}

    # collect loss
    def _collect_loss(self, canvas, run_feats, oracle_specs, oracle_scores, soft_trg_t=None):
        conf: ZDecoderIlmConf = self.conf
        ret_losses = []
        curriculum = self.train_curriculum.value
        # --
        left_ts, right_ts, self_ts, mask_ts = \
            canvas.history.left_ts, canvas.history.right_ts, canvas.history.self_ts, canvas.history.mask_ts
        # --
        all_left, all_right, all_self, all_weight, all_feat = [], [], [], [], []
        for ii, one_feat in enumerate(run_feats):
            if one_feat is None:  # [*, ?, D]
                continue
            one_left, one_right, one_self, one_mask = left_ts[ii], right_ts[ii], self_ts[ii], mask_ts[ii]  # [*, ?]
            if conf.train_reweight_stage:
                one_weight = one_mask / one_mask.sum(-1, keepdims=True).clamp(min=1.)  # [*, ?]
            else:
                one_weight = one_mask
            one_weight = (one_weight * (curriculum ** ii)).to(nnutils.DEFAULT_FLOAT)  # [1, a, a^2, ...]
            # --
            all_left.append(one_left)
            all_right.append(one_right)
            all_self.append(one_self)
            all_weight.append(one_weight)
            all_feat.append(one_feat)
        # concat them and compress
        all_left_t0 = torch.cat(all_left, -1)  # [*, AC0]
        all_right_t0 = torch.cat(all_right, -1)  # [*, AC0]
        all_self_t0 = torch.cat(all_self, -1)  # [*, AC0]
        all_weight_t0 = torch.cat(all_weight, -1)  # [*, AC0]
        all_feat_t0 = torch.cat(all_feat, -2)  # [*, AC0, D]
        _c_idx, all_vmask_t = nnutils.mask2idx(all_weight_t0>0)  # [*, AC]
        _do_compress = all_vmask_t.shape[-1] < (all_weight_t0.shape[-1] * conf.train_compress_thresh)
        if _do_compress:
            _vmask_ti = all_vmask_t.to(nnutils.DEFAULT_INT)
            all_left_t = all_left_t0.gather(-1, _c_idx) * _vmask_ti  # [*, AC]
            all_right_t = all_right_t0.gather(-1, _c_idx) * _vmask_ti  # [*, AC]
            all_self_t = all_self_t0.gather(-1, _c_idx) * _vmask_ti  # [*, AC]
            all_weight_t = all_weight_t0.gather(-1, _c_idx) * all_vmask_t  # [*, AC]
            _arange_t = nnutils.get_arange_t(all_feat_t0.shape[0], 1)
            all_feat_t = all_feat_t0[_arange_t, _c_idx]  # [*, AC, D]
        else:
            all_left_t, all_right_t, all_self_t, all_weight_t, all_feat_t = \
                all_left_t0, all_right_t0, all_self_t0, all_weight_t0, all_feat_t0
        # --
        # gather loss
        special_ids_t = canvas.id_t.clone()  # [*, elen]
        special_ids_t[..., 0] = self.ztask.special_vocab.noi_token_id  # note: put NOI at idx0 for convenience!
        all_score_t = self.incr.run_score_head(all_feat_t)  # [*, AC, V]
        all_score_nll = - all_score_t.log_softmax(dim=-1)  # [*, AC, V]
        if soft_trg_t is not None:
            all_cand_nll = torch.matmul(all_score_nll, soft_trg_t.transpose(-1,-2))  # [*, AC, L]
        else:
            all_cand_nll = select_scores(all_score_nll, special_ids_t)  # [*, AC, L]
        # get cands
        _arange_t = nnutils.get_arange_t(special_ids_t.shape[-1], 0)  # [L]
        all_cand_masks = ((all_left_t.unsqueeze(-1) < _arange_t) & (_arange_t < all_right_t.unsqueeze(-1)))\
            .to(nnutils.DEFAULT_FLOAT)  # [*, AC, L], valid candidates
        all_is_noi = ((all_cand_masks.sum(-1)<=0.) & (all_weight_t>0.)).to(nnutils.DEFAULT_FLOAT)  # [*, AC]
        all_cand_masks[..., 0] = all_is_noi  # note: idx0 is special one: NOI!
        # really gather loss: [*, AC]
        all_loss_t = self._get_loss(all_cand_nll, all_cand_masks, oracle_specs, oracle_scores,
                                    seg_idxes=None, left_t=all_left_t, right_t=all_right_t)  # [*, AC]
        # label smoothing
        _ls = conf.label_smoothing
        if _ls > 0.:
            all_loss_t = (1.-_ls) * all_loss_t + _ls * all_cand_nll.mean(-1)  # [*, AC]
        # unlike
        _unlike_w = conf.loss_unlike
        if _unlike_w > 0.:
            # first collect all stages
            _stage_t = canvas.stage_t  # [*, L]
            # get all realizes: *[*, ?, L]
            all_realize = [((_stage_t<=ii) & (_stage_t>=0)).unsqueeze(-2).expand(-1, one_feat.shape[-2], -1)
                           for ii, one_feat in enumerate(run_feats) if one_feat is not None]
            all_realize_t = torch.cat(all_realize, -2)  # [*, AC?, L]
            if _do_compress:
                _arange_t = nnutils.get_arange_t(all_realize_t.shape[0], 1)  # [*, 1]
                all_realize_t = all_realize_t[_arange_t, _c_idx]  # [*, AC, L]
            # prepare all targets (realized - cands)
            all_prob_t = all_score_t.softmax(-1)  # [*, AC, V]
            all_exc_t = torch.zeros_like(all_prob_t)  # [*, AC, V]
            orig_id_t = canvas.id_t.unsqueeze(-2)  # [*, 1, L]
            # note: here 0 is a special one PAD
            _to_exc_id_t = orig_id_t * all_realize_t.to(nnutils.DEFAULT_INT)  # [*, AC, L], (0 as pad!!)
            _to_inc_id_t = orig_id_t * all_cand_masks.to(nnutils.DEFAULT_INT)  # [*, AC, L], (0 as pad!!)
            all_exc_t.scatter_(-1, _to_exc_id_t, 1.)
            all_exc_t.scatter_(-1, _to_inc_id_t, 0.)
            all_exc_t[..., 0] = 0.  # [*, AC, V], does not count the special PAD!
            # add the loss!
            _loss_unlike_t = - ((1 - (all_prob_t * all_exc_t).sum(-1)) + 1e-5).log()  # [*, AC], -log(1-sum(ps))
            # [NOPE] all_loss_t = all_loss_t + _loss_unlike_t * _unlike_w  # [*, AC]
            _loss_unlike_item = LossHelper.compile_leaf_loss(
                f'{self.name}_unlike', (_loss_unlike_t*all_weight_t).sum(), all_weight_t.sum(),
                loss_lambda=conf.loss_ilm*_unlike_w)
            ret_losses.append(_loss_unlike_item)
        # --
        # down-weight NOI?
        if conf.loss_noi_weight != 1.:
            _extra_weight = 1. - all_is_noi * (1.-conf.loss_noi_weight)  # [*, AC]
            all_weight_t = all_weight_t * _extra_weight  # [*, AC]
        # --
        _loss_item = LossHelper.compile_leaf_loss(
            f'{self.name}_nll', (all_loss_t*all_weight_t).sum(), all_weight_t.sum(), loss_lambda=conf.loss_ilm)
        ret_losses.append(_loss_item)
        return ret_losses

    # common routine for gather loss
    # [*, K, C], [*, K, C], object, [*, L], [*, K, C]
    # note: trg_mask_t[...,0] must mean NOI and should be 1. accordingly
    def _get_loss(self, nll_t, trg_mask_t, oracle_specs, oracle_scores,
                  seg_idxes=None, left_t=None, right_t=None, stage_t=None):
        conf: ZDecoderIlmConf = self.conf
        # --
        _shape = list(nll_t.shape)  # [*, K, C]
        # use different oracles
        stra_model, stra_bt, stra_ps, stra_alpha = oracle_specs
        if stra_model:  # model-based
            _oracle_t = - (nll_t.detach() if conf.oracle_model_detach else nll_t)
        elif stra_bt:  # on-the-fly balance-tree
            _arange_t = nnutils.get_arange_t(_shape[-1], 0)  # [C]
            _center = ((_arange_t * trg_mask_t).sum(-1) / trg_mask_t.sum(-1).clamp(min=1)).unsqueeze(-1)  # [*, k, 1]
            _oracle_t = - (_center - _arange_t).abs()  # [*, k, elen], distance to center
        elif stra_ps:  # pair-scoring
            # _oracle_t = oracle_scores.unsqueeze(-3)  # [*, 1, L, L]
            _oracle_t = oracle_scores.unsqueeze(1)  # [*, 1, ...], note: it may be 2d or 1d behind
        else:
            if seg_idxes is None:  # in this case, C==L
                _oracle_t = oracle_scores.unsqueeze(-2)  # [*, 1, L]
            else:  # further select
                _oracle_t = oracle_scores.gather(-1, seg_idxes.view(_shape[:-2]+[-1])).view(_shape)
        _oracle_t = _oracle_t * stra_alpha
        # post processing for _oracle
        _oracle_t = self.ps_helper.score(_oracle_t, trg_mask_t, left_t, right_t, stage_t)  # [*, k, L]
        # special alpha
        if conf.span_size_alpha > 0.:
            _oracle_t = _oracle_t / (trg_mask_t.sum(-1, keepdims=True).clamp(min=1) ** conf.span_size_alpha)
        # add -inf for invalid ones
        _oracle_t = _oracle_t + get_pad_values(trg_mask_t)  # [*, k, L]
        # scale by temperature tau
        _tau = self.oracle_tau.value
        if _tau == 0.:  # argmax
            # if conf.oracle_take_max:  # take max one for oracle
            _, _max_idxes = _oracle_t.max(-1, keepdims=True)  # [*, k, L]
            trg_weight_t = torch.zeros_like(_oracle_t)
            trg_weight_t.scatter_(-1, _max_idxes, 1.)  # [*, K, L]
        else:  # [*, K, C]
            trg_weight_t = (_oracle_t.log_softmax(-1) / _tau).softmax(-1)  # first norm to avoid overflow
        # mix uniform
        _mixu = max(0., min(1., self.oracle_mixu.value))
        if _mixu > 0.:
            _uniform_weight = trg_mask_t / trg_mask_t.sum(-1, keepdims=True).clamp(min=1.)  # [*, K, C]
            trg_weight_t = (1.-_mixu) * trg_weight_t + _mixu * _uniform_weight
        # --
        trg_weight_t = trg_weight_t * trg_mask_t  # remember the mask!!
        final_loss_t = (nll_t * trg_weight_t).sum(-1)  # [*, K]
        return final_loss_t  # [*, K]

    # --
    def do_predict(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderIlmConf = self.conf
        if conf.test_do_sample and conf.test_beam_size > conf.test_sample_max_bs:
            orig_bs = conf.test_beam_size
            conf.test_beam_size = conf.test_sample_max_bs
            ret = {}
            for _ in range((orig_bs + conf.test_sample_max_bs - 1) // conf.test_sample_max_bs):
                ret = self._do_predict(med, *args, **kwargs)  # run it multiple times
            conf.test_beam_size = orig_bs
            return ret
        else:
            return self._do_predict(med, *args, **kwargs)

    def _do_predict(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderIlmConf = self.conf
        # --
        # prepare
        bsize = len(med.ibatch.insts)
        beam_size = conf.test_beam_size
        cross_t, cross_mask_k, max_len_t = self.prepare_search(med, beam_size, conf.test_max_len, conf.test_max_ratio)
        # --
        # run search
        od_canvas = self.searcher.open_decoding(bsize, beam_size, max_len_t, conf.test_noi_penalty, conf.test_mid_div, conf.test_final_div, conf.test_record_history, conf.test_compress_thresh, close_only_all_noi=conf.close_only_all_noi, dtp=conf.test_dtp, cross_t=cross_t, cross_mask_k=cross_mask_k)
        # assign them: [bs*beam, L']
        arr_id, arr_mask, arr_stage, arr_score = \
            [nnutils.get_value(z) for z in [od_canvas.id_t, od_canvas.mask_t, od_canvas.stage_t, od_canvas.score_t]]
        _inst_setter = self.ztask._inst_setter
        _all_stage = 0
        for bidx, inst in enumerate(med.ibatch.insts):
            _ii = bidx * beam_size  # already sorted
            best_id, best_mask, best_stage = arr_id[_ii], arr_mask[_ii]>0., arr_stage[_ii]
            list_id, list_stage = best_id[best_mask].tolist(), best_stage[best_mask].tolist()
            list_token = self.tokenizer.convert_ids_to_tokens(list_id)
            _inst_setter(inst, list_token, list_id)
            inst.info.update({'stage': list_stage, 'num_stage': 0 if len(list_stage)<=0 else max(list_stage)})
            _all_stage += inst.info['num_stage']
            # --
            if conf.test_output_all:  # further export all ones!
                res = []
                for _jj in range(beam_size):
                    _this_id, _this_mask, _this_stage = arr_id[_ii+_jj], arr_mask[_ii+_jj] > 0., arr_stage[_ii+_jj]
                    _this_idL, _this_stageL = _this_id[_this_mask].tolist(), _this_stage[_this_mask].tolist()
                    _this_toks = self.tokenizer.convert_ids_to_tokens(_this_idL)
                    _this_score = arr_score[_ii + _jj].item()
                    res.append({"tok": _this_toks, "score": _this_score, "stages": _this_stageL})
                if 'res' not in inst.info:
                    inst.info['res'] = []
                inst.info['res'].extend(res)
            # --
        # --
        return {"all_stage": _all_stage}

# --
# b zgen/model/tasks/dec_ilm/base:??
