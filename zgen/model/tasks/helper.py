#

# some common helpers

__all__ = [
    "run_bert_tok", "eval_inslm", "Noiser", "NoiserConf",
]

from typing import List
import numpy as np
import torch
import torch.nn.functional as F

from zgen.utils import nn as nnutils
from zgen.utils import MathHelper, zlog, ResultRecord, Conf

# tokenize with bert's tokenizer
def run_bert_tok(tokenizer, tokens: List[str], do_subtok: bool, do_add_ends: bool):
    # prepare things as the format of bert input!
    if do_subtok:
        # todo(note): this can change seq-len!!
        idxes = tokenizer(tokens, is_pretokenized=True)['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(idxes)
    else:
        idxes = tokenizer.convert_tokens_to_ids(tokens)
        if do_add_ends:
            idxes = [tokenizer.cls_token_id] + idxes + [tokenizer.sep_token_id]
    return tokens, idxes

# eval for some cases
def eval_inslm(name: str, gold_insts, pred_insts, quite: bool):
    res = {'loss_all': 0., 'loss_tok': 0., 'loss_noi': 0., 'stages': 0,
           'count_all': 0, 'count_tok': 0, 'count_noi': 0, 'seqs': 0}
    for inst in pred_insts:
        one_stages, one_tok_scores, one_noi_scores = inst.info[f'{name}_info']
        res['loss_all'] -= sum(one_tok_scores) + sum(one_noi_scores)
        res['loss_tok'] -= sum(one_tok_scores)
        res['loss_noi'] -= sum(one_noi_scores)
        res['count_all'] += len(inst)-2  # note: here no inclusion of noi-tokens!
        res['count_tok'] += len(inst)-2
        res['count_noi'] += len(inst)-1
        res['stages'] += max(one_stages)
        res['seqs'] += 1
    for suffix in ['all', 'tok', 'noi']:
        res[f'avg_loss_{suffix}'] = MathHelper.safe_div(res[f'loss_{suffix}'], res[f'count_{suffix}'])
    res['avg_stage'] = MathHelper.safe_div(res['stages'], res['seqs'])
    res['ppl'] = np.exp(res['avg_loss_all']).item()
    if not quite:
        zlog(f"=>Result of {name}_eval: {res}")
    return ResultRecord(results=res, score=-res['avg_loss_all'])

# --
# helper to make noisy data
class NoiserConf(Conf):
    def __init__(self):
        self.mask_range = [0., 0.]  # [low, high)
        self.combine_mask = True  # combine continuous mask?
        self.delete_range = [0.2, 0.3]  # [low, high)
        # --

class Noiser:
    def __init__(self, conf: NoiserConf):
        self.conf = conf
        # --

    # [*, L]
    def obtain_noise(self, mask_t, kept_t):
        conf = self.conf
        _shape = list(mask_t.shape)
        # --
        app_tb = ((mask_t>0.) & (kept_t<=0.))  # [*,L], applicable?
        survive_tb, domask_tb = (mask_t>0.), (mask_t>100.)  # [*, L]
        # first do mask
        _mask_a, _mask_b = conf.mask_range
        if _mask_b >= _mask_a and _mask_b > 0.:
            mask_ratio = _mask_a + (_mask_b-_mask_a) * nnutils.rand(_shape[:-1]).unsqueeze(-1)  # [*, 1]
            domask_tb = ((nnutils.rand(_shape) < mask_ratio) & app_tb)  # [*,L]
            if conf.combine_mask:  # delete the following ones
                _tmp_del_tb = (domask_tb[...,:-1] & domask_tb[...,1:])  # [*, L-1]
                _tmp_del_tb1 = F.pad(_tmp_del_tb, [1,0], value=False)  # [*, 1+(L-1)]
                survive_tb &= (~_tmp_del_tb1)
        # then delete things
        _del_a, _del_b = conf.delete_range
        if _del_b >= _del_a and _del_b > 0.:
            del_ratio = _del_a + (_del_b-_del_a) * nnutils.rand(_shape[:-1]).unsqueeze(-1)  # [*, 1]
            dodel_tb = ((nnutils.rand(_shape) < del_ratio) & app_tb)  # [*,L]
            survive_tb &= (~dodel_tb)
        # --
        return survive_tb, domask_tb

    def apply_noise(self, survive_tb, domask_tb, id_t, mask_id, pad_id=0):
        # first do mask
        masked_id_t = torch.where(domask_tb, nnutils.input_idx(mask_id), id_t)  # [*, L]
        # then do del
        _idxes_t, _masks_t = nnutils.mask2idx(survive_tb)  # [*, ??]
        ret_id_t = masked_id_t.gather(-1, _idxes_t)  # [*, ??]
        ret_id_t[_masks_t<=0.] = pad_id  # pad by zero
        return ret_id_t, _masks_t  # [*, ??]

    def make_noise_data(self, id_t, mask_t, kept_t, mask_id, pad_id=0):
        survive_tb, domask_tb = self.obtain_noise(mask_t, kept_t)
        ret_id_t, _masks_t = self.apply_noise(survive_tb, domask_tb, id_t, mask_id, pad_id=pad_id)
        return ret_id_t, _masks_t

    def make_noise_data_keep_ends(self, id_t, mask_t, mask_id, pad_id=0):
        # simply keeping ends
        kept_t = torch.zeros_like(mask_t)  # [*, L]
        kept_t[..., 0] = 1.
        _lidx = mask_t.sum(-1, keepdims=True).long() - 1  # [*, 1]
        kept_t.scatter_(-1, _lidx, 1.)
        return self.make_noise_data(id_t, mask_t, kept_t, mask_id, pad_id=pad_id)

# --
# b zgen/model/tasks/helper:107
