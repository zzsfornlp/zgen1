#

# utilizing bert encoder

__all__ = [
    "ZTaskEncBertConf", "ZTaskEncBert", "ZEncoderBertConf", "ZEncoderBert",
]

from typing import List
from zgen.utils import nn as nnutils
from zgen.utils import ConfEntryChoices
from ..core import *
from .base import *
from .helper import Noiser, NoiserConf

# --

class ZTaskEncBertConf(ZTaskBaseTConf):
    def __init__(self):
        super().__init__()
        self.name = "enc"
        self.bert_conf = ZEncoderBertConf()

    def build_task(self):
        return ZTaskEncBert(self)

class ZTaskEncBert(ZTaskBaseT):
    def __init__(self, conf: ZTaskEncBertConf):
        super().__init__(conf)
        # --

    def build_mod(self, model):
        return self.conf.bert_conf.make_node(self, model)

class ZEncoderBertConf(ZModBaseTConf):
    def __init__(self):
        super().__init__()
        # --
        # put noise to inputs?
        self.noiser = ConfEntryChoices({"yes": NoiserConf(), "no": None}, "no")
        self.test_do_noise = False  # do noising at test time?
        # ways to constrict mem
        self.mem_compress_method = "nope"  # nope/idx0/avg
        self.mem_detach = False
        # --

@node_reg(ZEncoderBertConf)
class ZEncoderBert(ZModBaseT):
    def __init__(self, conf: ZEncoderBertConf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        # --
        self._compress_f = getattr(self, "_compress_"+conf.mem_compress_method)
        if conf.noiser is not None:
            self.noiser = Noiser(conf.noiser)
        else:
            self.noiser = None
        # --

    # --
    def do_prep(self, med: ZMediator, *args, **kwargs):
        # prepare input
        IDX_PAD = self.tokenizer.pad_token_id
        ids = self.get_inst_idxes(med.ibatch.insts)
        idxes_t, masks_t = nnutils.go_batch_2d(ids, IDX_PAD)
        # --
        if (self.noiser is not None) and (self.training or self.conf.test_do_noise):
            IDX_MASK = self.tokenizer.mask_token_id
            idxes_t, masks_t = self.noiser.make_noise_data_keep_ends(idxes_t, masks_t, IDX_MASK, pad_id=IDX_PAD)
        # --
        med.set_cache((self.name, 'input'), (idxes_t, masks_t))  # [*, L]
        med.set_cache((self.name, 'input_length'), masks_t.sum(-1))  # [*]
        # --

    def do_loss(self, med: ZMediator, *args, **kwargs):
        conf: ZEncoderBertConf = self.conf
        # get input
        idxes_t, masks_t = med.get_cache((self.name, 'input'))
        # forward
        bert_outputs = self.forward_bert_model(input_ids=idxes_t, self_mask_k=masks_t)
        # put results!
        for ii, vv in enumerate(bert_outputs[1]):
            med.set_cache((self.name, 'hid'), vv, app=True, app_info=ii)
        # put things as enc-mem for decoders
        enc_mem, enc_mask = self._compress_f(bert_outputs[0], masks_t)
        if conf.mem_detach:
            enc_mem = enc_mem.detach()
        med.set_cache((self.name, 'enc_mem'), enc_mem)
        med.set_cache((self.name, 'enc_mask'), enc_mask)
        # --
        ret_loss = LossHelper.combine_multiple_losses([])
        return ret_loss, {}

    # --
    def _compress_nope(self, hid_t, mask_t):
        return hid_t, mask_t

    def _compress_idx0(self, hid_t, mask_t):
        hid_t = hid_t[..., 0:1, :]  # [*, L, 1]
        mask_t = mask_t[..., 0:1]  # [*, 1]
        return hid_t, mask_t

    def _compress_avg(self, hid_t, mask_t):
        weight_t = mask_t / mask_t.sum(-1, keepdims=True).clamp(min=1)  # [*, 1]
        hid_t = (hid_t * weight_t.unsqueeze(-1)).sum(-2, keepdims=True)  # [*, 1, D]
        mask_t = (mask_t.sum(-1, keepdims=True)>0).to(mask_t.dtype)  # [*, 1]
        return hid_t, mask_t
    # --

# --
# b zgen/model/tasks/enc_bert:100
