#

# base task & module with a bert(transformer) mod

__all__ = [
    "ZTaskBaseTConf", "ZTaskBaseT", "ZModBaseTConf", "ZModBaseT",
]

from typing import List
from zgen.utils import nn as nnutils
from zgen.utils import ConfEntryChoices, ZObject, ResultRecord
from zgen.data import inst_obtain_getter_setter, SimpleVocab
from zgen.eval import SeqEvaler, SeqEvalerConf
from ..core import *
from ..mods import *
from .helper import *

class ZTaskBaseTConf(ZTaskConf):
    def __init__(self):
        super().__init__()
        # --
        # process data
        self.seq_name = 'plain'
        self.do_prep = True
        # self.do_build_vocab = False  # build self's vocab?
        self.do_add_ends = False  # add [CLS] and [SEP]
        self.do_subtok = False  # need to further subtok data? (and add [CLS] and [SEP])
        # special names
        self.noi_token_name = '[unused2]'  # no-insertion (close)
        self.pau_token_name = '[unused3]'  # no-operation (pause)
        # eval
        self.eval = ConfEntryChoices({"seq": SeqEvalerConf(), "nope": None}, "nope")
        # --

class ZTaskBaseT(ZTask):
    def __init__(self, conf: ZTaskBaseTConf):
        super().__init__(conf)
        conf: ZTaskBaseTConf = self.conf
        # --
        self._inst_getter, self._inst_setter = inst_obtain_getter_setter(conf.seq_name)
        self._special_vocab = None  # special vocab
        if conf.eval is not None:
            conf.eval.eval_seq_name = conf.seq_name
            self.evaler = SeqEvaler(conf.eval)
        else:
            self.evaler = None
        # --

    def eval_insts(self, gold_insts, pred_insts, quite=False):
        res = None
        if self.evaler is not None:
            res_dict = self.evaler.eval(gold_insts, pred_insts, quite=quite)
            res = ResultRecord(results=res_dict, score=res_dict['res'])
        return res

    def build_vocab(self, datasets: List):
        conf: ZTaskBaseTConf = self.conf
        # --
        # note: build externally!!
        # if conf.do_build_vocab:
        #     voc = SimpleVocab.build_empty(
        #         self.name, pre_list=("pad", "unk", "eos", "bos", "mask", "noi", "pau", "s1", "s2", "s3"), post_list=())
        #     for data in datasets:
        #         for inst in data.yield_insts():
        #             voc.feed_iter(self._inst_getter(inst)[0])
        #     voc.build_sort()
        #     voc.feed_one(conf.noi_token_name)
        #     voc.feed_one(conf.pau_token_name)
        #     voc.set_pre_post()
        #     return ZTokerVWrapper(voc)
        return None  # note: currently simply use pre-trained vocabs!

    def prep_f(self, inst, dataset):
        conf: ZTaskBaseTConf = self.conf
        # --
        if conf.do_prep:  # if need to prepare
            bert_toker = self.mod.tokenizer
            tok_res = run_bert_tok(bert_toker, self._inst_getter(inst)[0], conf.do_subtok, conf.do_add_ends)
            self._inst_setter(inst, *tok_res)
        return inst
        # --

    @property
    def special_vocab(self):
        toker = self.mod.tokenizer
        # --
        def _get_id(_name: str):
            _id = toker.convert_tokens_to_ids([_name])[0]
            assert toker.convert_ids_to_tokens([_id]) == [_name]
            return _id
        # --
        if self._special_vocab is None:
            conf: ZTaskBaseTConf = self.conf
            noi_token_id, pau_token_id = _get_id(conf.noi_token_name), _get_id(conf.pau_token_name)
            self._special_vocab = ZObject(
                cls_token_id=toker.cls_token_id, sep_token_id=toker.sep_token_id, pad_token_id=toker.pad_token_id,
                mask_token_id=toker.mask_token_id, noi_token_id=noi_token_id, pau_token_id=pau_token_id)
        return self._special_vocab

class ZModBaseTConf(ZModConf):
    def __init__(self):
        super().__init__()
        # --
        self.bconf = ConfEntryChoices({"own": ZBertConf(), "shared": None}, "own")
        self.shared_bert_mname = ""  # if bert is shared from others, the name of that mod!
        self.bert_ft = True  # whether fine-tune the model
        self.cross_mname = ''  # src tensor mod name
        self.label_smoothing = 0.
        # --

@node_reg(ZModBaseTConf)
class ZModBaseT(ZMod):
    def __init__(self, conf: ZModBaseTConf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, **kwargs)
        # --
        conf: ZModBaseTConf = self.conf
        if conf.bconf is None:
            _mod = zmodel.get_mod(conf.shared_bert_mname)
            assert isinstance(_mod, ZModBaseT)
            _bert = _mod.bert
        else:
            if conf.cross_mname:
                _src_mod = zmodel.get_mod(conf.cross_mname)
                add_cross_att, cross_att_size = True, _src_mod.bert.hidden_size
            else:
                add_cross_att, cross_att_size = False, -1
            my_toker = ztask.vpack
            if my_toker is not None:  # wrap it to be similar to bert's toker!
                my_toker = ZTokerVWrapper(my_toker)
            _bert = ZBert(conf.bconf, external_toker=my_toker, add_cross_att=add_cross_att, cross_att_size=cross_att_size)
        if conf.bert_ft:
            self.bert = _bert
        else:  # no add module!!
            self.setattr_borrow('bert', _bert)
        # --

    @property
    def tokenizer(self):
        return self.bert.tokenizer

    # common helpers
    def get_inst_idxes(self, insts):
        ff = self.ztask._inst_getter
        return [ff(inst)[1] for inst in insts]

    def forward_bert_model(self, *args, **kwargs):
        with nnutils.no_grad_env(no_grad=(not self.conf.bert_ft)):
            bert_outputs = self.bert.forward_model(*args, **kwargs)
        return bert_outputs

    # prepare for beam search
    def prepare_search(self, med, parallel_size: int, max_len: int, max_ratio: float):
        cross_mname = self.conf.cross_mname
        bsize = len(med.ibatch.insts)
        if cross_mname:  # [bs, L, D?]
            cross_t, cross_mask_k = med.get_cache((cross_mname, 'enc_mem')), med.get_cache((cross_mname, 'enc_mask'))
            # src length
            src_input_len_t = med.get_cache((cross_mname, 'input_length'), [max_len] * bsize)  # [bs]
            max_len_t = (nnutils.input_real(src_input_len_t) * max_ratio).to(nnutils.DEFAULT_INT)  # [bs]
            max_len_t.clamp_(max=max_len)
            if parallel_size != 1:  # note: expand at the outside for beam: [bs*beam, L, D?]
                cross_t, cross_mask_k = cross_t.repeat_interleave(parallel_size, dim=0), \
                                        cross_mask_k.repeat_interleave(parallel_size, dim=0)
                max_len_t = max_len_t.repeat_interleave(parallel_size, dim=0)  # [bs*beam]
        else:
            cross_t, cross_mask_k = None, None
            max_len_t = max_len
        return cross_t, cross_mask_k, max_len_t  # [bs*beam, ?]

# --
# b zgen/model/tasks/base:74
