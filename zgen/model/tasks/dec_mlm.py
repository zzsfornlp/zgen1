#

# MLM

__all__ = [
    "ZTaskMlmConf", "ZTaskMlm", "ZDecoderMlmConf", "ZDecoderMlm",
]

from typing import List
import numpy as np
from copy import deepcopy
from collections import Counter
import torch
from zgen.utils import ResultRecord, MathHelper, zlog
from zgen.utils import nn as nnutils
from ..core import *
from .base import *

# --

class ZTaskMlmConf(ZTaskBaseTConf):
    def __init__(self):
        super().__init__()
        self.name = "mlm"
        self.mlm_conf = ZDecoderMlmConf()

    def build_task(self):
        return ZTaskMlm(self)

class ZTaskMlm(ZTaskBaseT):
    def __init__(self, conf: ZTaskMlmConf):
        super().__init__(conf)
        # --

    def build_mod(self, model):
        return self.conf.mlm_conf.make_node(self, model)

    def eval_insts(self, gold_insts, pred_insts, quite=False):
        # no need of gold!
        res = {'loss': 0., 'toks': 0, 'count': 0, 'corr1': 0, 'corrA': 0}
        for inst in pred_insts:
            res['toks'] += len(inst)
            for tidx, item in enumerate(inst.info['mlm_info']):
                if item is not None:
                    cur_tok = inst.tokens[tidx]
                    res['loss'] += item['loss']
                    res['count'] += 1
                    res['corr1'] += int(item['topk_toks'][0] == cur_tok)
                    res['corrA'] += int(cur_tok in item['topk_toks'])
        res['avg_loss'] = MathHelper.safe_div(res['loss'], res['count'])
        res['acc1'] = MathHelper.safe_div(res['corr1'], res['count'])
        res['accA'] = MathHelper.safe_div(res['corrA'], res['count'])
        if not quite:
            zlog(f"=>Result of mlm_eval: {res}")
        return ResultRecord(results=res, score=-res['avg_loss'])

# --

class ZDecoderMlmConf(ZModBaseTConf):
    def __init__(self):
        super().__init__()
        # --
        self.loss_mlm = 1.
        # mlm specific
        self.mlm_mrate = 0.15  # how much to mask?
        self.mlm_repl_rates = [0.8, 0.1, 0.1]  # rates of: [MASK], random, unchanged
        # --

    def get_repl_ranges(self):
        _arr = np.asarray(self.mlm_repl_rates)
        _a, _b, _c = (_arr/_arr.sum()).cumsum().tolist()
        assert _c == 1.
        return _a, _b

@node_reg(ZDecoderMlmConf)
class ZDecoderMlm(ZModBaseT):
    def __init__(self, conf: ZDecoderMlmConf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZDecoderMlmConf = self.conf
        # --
        # note: specific ones!!
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        # --
        self.repl_ranges = conf.get_repl_ranges()
        self.target_size = int(self.bert.vocab_size)
        # --

    def do_prep(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderMlmConf = self.conf
        # --
        IDX_PAD = self.tokenizer.pad_token_id
        ids = self.get_inst_idxes(med.ibatch.insts)
        enc_ids, enc_mask = nnutils.go_batch_2d(ids, IDX_PAD)
        _shape = enc_ids.shape
        # sample mask
        mlm_mask = ((nnutils.rand(_shape) < conf.mlm_mrate) & (enc_ids != self.cls_token_id) &
                    (enc_ids != self.sep_token_id)).to(nnutils.DEFAULT_FLOAT) * enc_mask  # [*, elen]
        # sample repl
        _repl_sample = nnutils.rand(_shape)  # [*, elen], between [0, 1)
        mlm_repl_ids = nnutils.constants(_shape, self.mask_token_id, dtype=nnutils.DEFAULT_INT)  # [*, elen] [MASK]
        _repl_rand, _repl_origin = self.repl_ranges
        mlm_repl_ids = torch.where(_repl_sample>_repl_rand,
                                   (nnutils.rand(_shape, dtype=torch.float32)*self.target_size).long(), mlm_repl_ids)
        mlm_repl_ids = torch.where(_repl_sample>_repl_origin, enc_ids, mlm_repl_ids)
        # final prepare
        mlm_input_ids = torch.where(mlm_mask>0., mlm_repl_ids, enc_ids)  # [*, elen]
        # --
        med.set_cache((self.name, 'info'), (enc_ids, mlm_mask))  # orig ids and mlm masks
        med.set_cache((self.name, 'input'), (mlm_input_ids, enc_mask), assert_nonexist=False)  # directly replace!
        # --

    def calc_output(self, med):
        # forward
        idxes_t, masks_t = med.get_cache((self.name, 'input'))
        bert_outputs = self.forward_bert_model(input_ids=idxes_t, self_mask_k=masks_t)
        # get final output
        hid = bert_outputs[0]  # [*, elen, D]
        # select according to mask
        enc_ids, mlm_mask = med.get_cache((self.name, 'info'))
        valid_t = (mlm_mask > 0.)
        flatten_hid = hid[valid_t]  # [??, D]
        flatten_ids = enc_ids[valid_t]  # [??]
        flatten_output = self.bert.forward_lmhead(flatten_hid)  # [??, L]
        return mlm_mask, flatten_ids, flatten_output

    def do_loss(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderMlmConf = self.conf
        # --
        loss_items = []
        if conf.loss_mlm > 0.:
            mlm_mask, flatten_ids, flatten_output = self.calc_output(med)
            loss_t = nnutils.loss_nll(flatten_output, flatten_ids, label_smoothing=conf.label_smoothing)  # [??]
            # --
            _loss_item = LossHelper.compile_leaf_loss(
                f'{self.name}_nll', loss_t.sum(), torch.ones_like(loss_t).sum(), loss_lambda=conf.loss_mlm)
            loss_items.append(_loss_item)
        # --
        ret_loss = LossHelper.combine_multiple_losses(loss_items)
        return ret_loss, {}

    def do_predict(self, med: ZMediator, *args, **kwargs):
        tokenizer = self.tokenizer
        # --
        mlm_mask, flatten_ids, flatten_output = self.calc_output(med)
        loss_t = nnutils.loss_nll(flatten_output, flatten_ids)  # [??]
        prob_t = flatten_output.softmax(-1)  # [??, V]
        topk_probs_t, topk_idxes_t = prob_t.topk(5, dim=-1)  # [??, K]
        # assign output
        arr_mlm_mask, arr_loss, arr_topk_probs, arr_topk_idxes = \
            [nnutils.get_value(z) for z in (mlm_mask, loss_t, topk_probs_t, topk_idxes_t)]
        ii = 0
        for bidx, inst in enumerate(med.ibatch.insts):
            mlm_info = [None] * len(inst)
            for tidx, mm in enumerate(arr_mlm_mask[bidx]):
                if mm > 0.:
                    mlm_info[tidx] = {
                        'loss': arr_loss[ii].item(), 'topk_toks': tokenizer.convert_ids_to_tokens(arr_topk_idxes[ii]),
                        'topk_probs': arr_topk_probs[ii].tolist()}
                    ii += 1
            inst.info['mlm_info'] = mlm_info
        assert ii == len(arr_loss)
        # --
        return {}
