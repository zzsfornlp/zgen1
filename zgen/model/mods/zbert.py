#

# implement according to "transformers.modeling_bert"
# -- with extensions of easier src-condition and seg-incremental-encoding/decoding

__all__ = [
    "ZBertConf", "ZBert", "BertAttention"
]

from typing import Dict, List
import torch
import math
import re
from copy import deepcopy
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
from zgen.utils import zwarn, zlog
from zgen.utils import nn as nnutils
from ..core import *

# --
class ZBertConf(ZNodeConf):
    def __init__(self):
        self.config = None  # to be added from pre_trained points!!
        self.zbert_model = "bert-base-multilingual-cased"  # or "bert-base-multilingual-cased", "bert-base-cased", "bert-large-cased", "bert-base-chinese", "roberta-base", "roberta-large", "xlm-roberta-base", "xlm-roberta-large", "distilbert-base-cased",
        self.zbert_cache_dir = ""  # dir for downloading
        self.zbert_no_pretrain = False  # no init from pretrained
        self.zbert_kwargs = {}
        self.zbert_add_lmhead = False  # add an extra output layer
        # --
        # for embeddings
        self.no_posi_embeds = False  # no position embeddings!
        self.no_type_embeds = False  # no type embeddings (this is true for some pretrained models!)
        self.posi_embed_offset = 0  # offset for positional embeddings
        # for selfatt
        self.self_att_relC = 0  # relative position clip-k, '==0' means off
        # for crossattention
        self.add_cross_att = False  # adopt extra src input and add cross-att after self-att
        self.cross_att_size = -1  # if <=0, then the same as self's dimension
        # output
        self.lmhead_no_hid = False
        self.lmhead_tie_embeds = False
        # --
        # special setup
        self.zbert_extra_setup = ""
        # --

    @property
    def cache_dir_or_none(self):
        return self.zbert_cache_dir if self.zbert_cache_dir else None

# lazy loader!
class ZBertHelper:
    # --
    INFO_CENTER = {
        "bert": {
            "path_bert": "bert".split(), "path_lmhead": "cls".split(),
            "subs_bert": [("LayerNorm", "layer_norm"), (r"attention\.self", "attention"), (r"attention\.output", "attention"), (r"output\.dense", "output.dense2"), ("intermediate", "ffn"), ("output", "ffn")] + [(z, "") for z in ["^pooler.*", "embeddings\.position_ids"]],
            "subs_lmhead": [(r"predictions\.bias", ""), (r"predictions\.", ""), (r"transform\.", ""), ("LayerNorm", "layer_norm"), ],
            "maps_config": {},  # no need to map for bert itself
            "zconf": {},
        },
        "roberta": {
            "path_bert": "roberta".split(), "path_lmhead": "lm_head".split(),
            "subs_bert": [("LayerNorm", "layer_norm"), (r"attention\.self", "attention"), (r"attention\.output", "attention"), (r"output\.dense", "output.dense2"), ("intermediate", "ffn"), ("output", "ffn")] + [(z, "") for z in ["^pooler.*", "embeddings\.position_ids"]],
            "subs_lmhead": [(r"^bias", ""), ],
            # note: a trick to make it feasible !!
            "maps_config": {'type_vocab_size': 1, 'layer_norm_eps': 1e-5, 'max_position_embeddings': 512},
            "zconf": {'posi_embed_offset': 2},
        },
        "distilbert": {
            "path_bert": "distilbert".split(), "path_lmhead": [],
            "subs_bert": [("LayerNorm", "layer_norm"), ("transformer", "encoder"), ("q_lin", "query"), ("k_lin", "key"), ("v_lin", "value"), ("out_lin", "dense"), ("sa_layer_norm", "attention.layer_norm"), (r"ffn\.lin1", "ffn.dense"), (r"ffn\.lin2", "ffn.dense2"), ("output_layer_norm", "ffn.layer_norm")],
            "subs_lmhead": [("^distilbert.*", ""), ("vocab_transform", "dense"), ("vocab_layer_norm", "layer_norm"), ("vocab_projector", "decoder")],
            "maps_config": {'intermediate_size': 'hidden_dim', 'hidden_act': 'activation', 'hidden_dropout_prob': 'dropout', 'attention_probs_dropout_prob': 'attention_dropout', 'type_vocab_size': 0, 'layer_norm_eps': 1e-12},
            "zconf": {},
        },
        # bart: separate encoder and decoder
        # note: 'activation_dropout' is not used
        "bartE": {
            "path_bert": "model.encoder".split('.'), "path_lmhead": None,
            "subs_bert": [("embed_tokens", "embeddings.word_embeddings"), ("embed_positions", "embeddings.position_embeddings"), ("layernorm_embedding", "embeddings.layer_norm"), ("layers", "encoder.layer"), (r"self_attn\.", "attention."), ("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value"), ("out_proj", "dense"), ("self_attn_layer_norm", "attention.layer_norm"), ("fc1", "ffn.dense"), ("fc2", "ffn.dense2"), ("final_layer_norm", "ffn.layer_norm")],
            "subs_lmhead": [],
            "maps_config": {'hidden_size': 'd_model', 'num_hidden_layers': 'encoder_layers', 'num_attention_heads': 'encoder_attention_heads', 'intermediate_size': 'encoder_ffn_dim', 'hidden_act': 'activation_function', 'hidden_dropout_prob': 'dropout', 'attention_probs_dropout_prob': 'attention_dropout', 'type_vocab_size': 0, 'initializer_range': 'init_std', 'layer_norm_eps': 1e-5},
            "zconf": {'posi_embed_offset': 2, 'lmhead_no_hid': True, 'lmhead_tie_embeds': True},
        },
        "bartD": {
            "path_bert": "model.decoder".split('.'), "path_lmhead": None,
            "subs_bert": [("embed_tokens", "embeddings.word_embeddings"), ("embed_positions", "embeddings.position_embeddings"), ("layernorm_embedding", "embeddings.layer_norm"), ("layers", "encoder.layer"), (r"self_attn\.", "attention."), ("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value"), ("out_proj", "dense"), ("self_attn_layer_norm", "attention.layer_norm"), ("fc1", "ffn.dense"), ("fc2", "ffn.dense2"), ("final_layer_norm", "ffn.layer_norm")] + [(r"encoder_attn\.", "crossattention."), ("encoder_attn_layer_norm", "crossattention.layer_norm")],
            "subs_lmhead": [],
            "maps_config": {'hidden_size': 'd_model', 'num_hidden_layers': 'decoder_layers', 'num_attention_heads': 'decoder_attention_heads', 'intermediate_size': 'decoder_ffn_dim', 'hidden_act': 'activation_function', 'hidden_dropout_prob': 'dropout', 'attention_probs_dropout_prob': 'attention_dropout', 'type_vocab_size': 0, 'initializer_range': 'init_std', 'layer_norm_eps': 1e-5},
            "zconf": {'posi_embed_offset': 2, 'lmhead_no_hid': True, 'lmhead_tie_embeds': True, 'add_cross_att': True},
        },
    }
    # --
    EXTRA_INFO = {  # extra settings
        # when ilm, tie enc-dec, no tie lmhead, vocab 29k -> 60.3M / 210.3M
        "baseT": {
            "config": {'num_hidden_layers': 6, 'hidden_size': 512, 'intermediate_size': 2048,
                       'num_attention_heads': 8, 'hidden_dropout_prob': 0.1},
            "zconf": {},
        },
        "largeT": {
            "config": {'num_hidden_layers': 6, 'hidden_size': 1024, 'intermediate_size': 4096,
                       'num_attention_heads': 16, 'hidden_dropout_prob': 0.3},
            "zconf": {},
        },
        # when mt(en-de), vocab 40k -> 16.4M / 38.2M / 66.6M
        "tinyT": {
            "config": {'num_hidden_layers': 3, 'hidden_size': 256, 'intermediate_size': 1024,
                       'num_attention_heads': 4, 'hidden_dropout_prob': 0.1},
            "zconf": {},
        },
        "smallT": {
            "config": {'num_hidden_layers': 3, 'hidden_size': 512, 'intermediate_size': 1024,
                       'num_attention_heads': 8, 'hidden_dropout_prob': 0.1},
            "zconf": {},
        },
    }
    # --

    def __init__(self, conf: ZBertConf, external_toker=None, **kwargs):
        self.conf = ZBertConf.direct_conf(deepcopy(conf), **kwargs)
        conf = self.conf
        # --
        self.model_type, self.model_name, self.model_comps = self.parse_model_name(conf.zbert_model)
        self.model_info = ZBertHelper.INFO_CENTER[self.model_type]
        self.extra_info = ZBertHelper.EXTRA_INFO.get(conf.zbert_extra_setup)
        self._config = None
        if external_toker is None:
            self._tokenizer = None  # load from pre-trained
        else:
            self._tokenizer = external_toker
            conf.zbert_kwargs['vocab_size'] = external_toker.vocab_size  # change here!
            zlog(f"Use exteranl_toker: {external_toker}!!")
        self._bert = None
        self._lmhead = None
        # --

    @property
    def config(self):
        if self._config is None:
            from transformers import AutoConfig
            conf = self.conf
            _config = AutoConfig.from_pretrained(self.model_name, cache_dir=conf.cache_dir_or_none)
            # --
            # map config
            from transformers import BertConfig
            bert_config = BertConfig()
            # first assign same name
            for k, v in bert_config.__dict__.items():
                if hasattr(_config, k):
                    setattr(bert_config, k, getattr(_config, k))
            # then assign renamed ones
            for trg, src in self.model_info['maps_config'].items():
                assert hasattr(bert_config, trg)
                if isinstance(src, str):
                    setattr(bert_config, trg, getattr(_config, src))
                else:  # otherwise directly set!
                    setattr(bert_config, trg, src)
            # --
            zbert_kwargs = {}
            # -- note: extra setup for config
            if self.extra_info is not None:
                zbert_kwargs.update(self.extra_info['config'])
            zbert_kwargs.update(conf.zbert_kwargs)  # later update this!
            # --
            for k, v in zbert_kwargs.items():
                assert hasattr(bert_config, k)
                setattr(bert_config, k, v)
            zlog(f"Zbert: load config {self.model_name} with {zbert_kwargs}: zbert.config={bert_config}")
            self._config = bert_config
        return self._config

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            conf = self.conf
            bert_toker = AutoTokenizer.from_pretrained(self.model_name, cache_dir=conf.cache_dir_or_none)
            zlog(f"Zbert: load tokenizer {self.model_name}: {bert_toker}")
            self._tokenizer = bert_toker
        return self._tokenizer

    def _load_model(self):
        from transformers import AutoModelForMaskedLM
        conf = self.conf
        bert_model = AutoModelForMaskedLM.from_pretrained(self.model_name, cache_dir=conf.cache_dir_or_none)
        _bert = self.retrieve(bert_model, self.model_info['path_bert'])
        _lmhead = None
        if self.model_info['path_lmhead'] is not None:
            _lmhead = self.retrieve(bert_model, self.model_info['path_lmhead'])
        else:
            zwarn(f"No lmhead for {self.model_name}!!")
        zlog(f"Zbert: load model {self.model_name}->{self.model_comps}.")
        return _bert, _lmhead

    @property
    def bert(self):
        if self._bert is None:
            self._bert, self._lmhead = self._load_model()
        return self._bert

    @property
    def lmhead(self):
        if self._bert is None:  # note: sometimes _lmhead can actually be None!
            self._bert, self._lmhead = self._load_model()
        return self._lmhead

    def parse_model_name(self, zbert_model: str):
        model_name, *model_comps = zbert_model.split(".")
        model_type = model_name.split("/")[-1].split("-")[0]  # bert,bart,xlm,...
        if model_type == "bart":
            assert len(model_comps)==1 and model_comps[0] in ["encoder", "decoder"]
            model_type = model_type + str.upper(model_comps[0][0])
        if model_type == "distilroberta":
            model_type = "roberta"
        return model_type, model_name, model_comps

    def retrieve(self, m, path: List[str]):
        ret = m
        for n in path:
            ret = getattr(ret, n)
        return ret

    def load_state_dict(self, m, d):
        try:
            m.load_state_dict(d, strict=True)
        except:
            import traceback
            zlog(f"#== Error in strict loading:\n{traceback.format_exc()}\n#==")
            m.load_state_dict(d, strict=False)
        # --

    def rename_state_dict(self, d, subs):
        # simply loop for each sub!
        ret = d
        for s_pat, s_repl in subs:
            _new = OrderedDict()
            for k, v in ret.items():
                k2 = re.sub(s_pat, s_repl, k)
                if len(k2)>0:   # simply delete empty ones!
                    assert k2 not in _new
                    _new[k2] = v
            ret = _new
        return ret

    def setup_zconf(self, zconf: ZBertConf):
        config = self.config
        zconf.config = config
        _toset = self.model_info['zconf'].copy()
        # -- note: extra setup for zconf
        if self.extra_info is not None:
            _toset.update(self.extra_info['zconf'])
        # --
        if _toset:
            zlog(f"Force set zconf: {_toset}")
        for k, v in _toset.items():
            assert hasattr(zconf, k)
            setattr(zconf, k, v)
        # --

    def init_model(self, bert, lmhead):
        # first init bert
        _bert_dict = self.bert.state_dict()
        _bert_dict1 = self.rename_state_dict(_bert_dict, self.model_info['subs_bert'])
        self.load_state_dict(bert, _bert_dict1)
        # then init lmhead
        if lmhead is not None and self.lmhead is not None:
            _lmhead_dict = self.lmhead.state_dict()
            _lmhead_dict1 = self.rename_state_dict(_lmhead_dict, self.model_info['subs_lmhead'])
            self.load_state_dict(lmhead, _lmhead_dict1)
        # --

@node_reg(ZBertConf)
class ZBert(ZNode):
    # note: usually do not call this!
    def __init__(self, conf: ZBertConf, external_toker=None, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZBertConf = self.conf
        # --
        bert_helper = ZBertHelper(conf, external_toker=external_toker)
        # setup config & toker
        bert_helper.setup_zconf(conf)
        self.tokenizer = bert_helper.tokenizer
        # get a model
        self.model = BertModel(conf)
        if conf.zbert_add_lmhead:  # if additional LMHead
            self.lmhead = BertLMHead(conf)
        else:
            self.lmhead = None
        # init from pretrain?
        self.apply(self._init_weights)
        if not conf.zbert_no_pretrain:
            bert_helper.init_model(self.model, self.lmhead)
        else:
            zwarn("No pretrain-loading for bert, really want this?")
        # --
        # finally move to target
        zlog(f"Prepare ok, move to default device {nnutils.DEFAULT_DEVICE}")
        self.eval()  # note: by default set eval!!
        self.to(nnutils.DEFAULT_DEVICE)
        zlog("Move ok!")
        # --

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.conf.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        if self.conf.lmhead_tie_embeds:
            if self.lmhead is not None:
                assert self.lmhead.decoder.weight.shape == self.model.embeddings.word_embeddings.weight.shape
                # note: directly assign will be fine!!
                zlog(f"Tie embeds: {self.lmhead.decoder} <-> {self.model.embeddings.word_embeddings}")
                self.lmhead.decoder.weight = self.model.embeddings.word_embeddings.weight
        # --

    # --
    # various forward methods

    # how to run this:
    # 1) run full: input_ids, self_mask_k, [opt]self_mask_qk
    # 2) run incr: input_ids([S]), self_mask_k([S]), [opt]self_mask_qk([S,Prev+S]), cache
    # option extra_embeds: adding extra_embeds to the inputs
    # option cross_*: cross_t, cross_mask_k
    def forward_model(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward_lmhead(self, hid_t):
        return self.lmhead(hid_t)

    # some properties
    @property
    def hidden_size(self): return self.conf.config.hidden_size

    @property
    def vocab_size(self): return self.conf.config.vocab_size

    @property
    def num_layers(self): return self.conf.config.num_hidden_layers

    @property
    def num_heads(self): return self.conf.config.num_attention_heads

    @property
    def has_posi_embeds(self): return not self.conf.no_posi_embeds
    # --

# --
# components; note: mostly adopted from "transformers.modeling_bert"

class BertEmbeddings(nn.Module):
    def __init__(self, conf: ZBertConf):
        super().__init__()
        config = conf.config
        # --
        # todo(+N): this is a small bug, if no adds, pad0 will cause NAN -> a specific fix!
        if conf.no_posi_embeds and (conf.no_type_embeds or config.type_vocab_size==0):  # if no other embeddings
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        if not conf.no_posi_embeds:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings+conf.posi_embed_offset, config.hidden_size)
        else:
            self.position_embeddings = None
        self.posi_embed_offset = conf.posi_embed_offset
        if not conf.no_type_embeds and config.type_vocab_size>0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        else:
            self.token_type_embeddings = None
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, extra_embeds=None, cache=None):
        # first tok-id embeddings!
        cur_embeds = self.word_embeddings(input_ids)  # [*, len, D]
        # add position embeddings
        if self.position_embeddings is not None:
            _step = 0 if cache is None else cache.get('step', 0)
            if position_ids is None:  # by default, simply range(len)
                position_ids = nnutils.arange(input_ids.size(-1), dtype=nnutils.DEFAULT_INT) + _step
            if cache is not None:
                cache['step'] = _step + position_ids.size(-1)
            position_embeddings = self.position_embeddings(position_ids + self.posi_embed_offset)
            cur_embeds = cur_embeds + position_embeddings
        # note: sometimes we may want this (unordered encoder)
        # elif extra_embeds is None:  # in this case, should use extra_embeds instead!!
        #     zwarn("It seems no position information!!")
        # add type embeddings
        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)  # by default zeros
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            cur_embeds = cur_embeds + token_type_embeddings
        # add extra
        if extra_embeds is not None:
            cur_embeds = cur_embeds + extra_embeds
        # output
        cur_embeds = self.layer_norm(cur_embeds)
        cur_embeds = self.dropout(cur_embeds)
        return cur_embeds

class BertAttention(nn.Module):
    def __init__(self, conf: ZBertConf, kv_size=-1, kv_static=False, relC=0):
        super().__init__()
        config = conf.config
        # --
        # specify some dimensions
        assert not(config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # self-att
        if kv_size <= 0:  # <=0 means same as self
            kv_size = config.hidden_size
        self.kv_static = kv_static  # for cross-att, kv can be static and reuse through cache
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(kv_size, self.all_head_size)
        self.value = nn.Linear(kv_size, self.all_head_size)
        self.dropout0 = nn.Dropout(config.attention_probs_dropout_prob)
        # self-output
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        # --
        # relative position
        self.relC = relC
        if relC > 0:
            self.edge_atts = nn.Embedding(relC*2+1, self.attention_head_size)
            self.edge_values = nn.Embedding(relC*2+1, self.attention_head_size)
        else:
            self.edge_atts = self.edge_values = None
        # --

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # *[*, L?, D], [*, LK], [*, Q, PREV+LK]; cache: {k:??, v:??};; rposi: [*, Q, PREV+LK]
    def forward(self, query, key, value, mask_k=None, mask_qk=None, cache: Dict = None, rposi=None):
        _NEG_INF = -10000.
        # --
        # projection: [*, L?, D*H] + transpose: [*, H, L?, D]
        # query
        mixed_query_layer = self.query(query)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # key & value
        has_cache = cache is not None and len(cache)>0
        if has_cache and self.kv_static:  # already calculated the static cache, directly reuse!
            key_layer, value_layer, mask_k = cache['k'], cache['v'], cache['m']
        else:  # still need to calculate first
            # key
            mixed_key_layer = self.key(key)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            # value
            mixed_value_layer = self.value(value)
            value_layer = self.transpose_for_scores(mixed_value_layer)
            # mask
            if mask_k is None:  # all 1s
                mask_k = nnutils.constants(key.shape[:-1], value=1., dtype=nnutils.DEFAULT_FLOAT)  # [*, LK]
            # --
            if has_cache and not self.kv_static:  # concat previous ones!
                key_layer = torch.cat([cache['k'], key_layer], -2)  # [*, H, PREV+LK, D]
                value_layer = torch.cat([cache['v'], value_layer], -2)  # [*, H, PREV+LK, D]
                mask_k = torch.cat([cache['m'], mask_k], -1)  # [*, PREV+LK]
        # --
        # set cache: directly replace!
        if cache is not None:
            cache['k'] = key_layer
            cache['v'] = value_layer
            cache['m'] = mask_k
        # --
        # dot-product: [*, H, Q, K]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # --
        # relative
        if rposi is not None:
            rposi = rposi.clamp(-self.relC, self.relC) + self.relC  # clip
        if rposi is not None:
            _query2 = torch.matmul(query_layer, self.edge_atts.weight.T)  # [*, H, Q, 2C+1]
            _rscore = _query2.gather(-1, rposi.unsqueeze(-3).expand(-1, _query2.size()[1], -1, -1))  # [*, H, Q, K]
            attention_scores = attention_scores + _rscore
        # --
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # mask
        att_mask = mask_k.unsqueeze(-2)  # [*, 1, K]
        if mask_qk is not None:
            att_mask = att_mask * mask_qk  # [*, Q, K]
        attention_scores = attention_scores + (1.-att_mask).unsqueeze(-3) * _NEG_INF  # [*, H, Q, K]
        # Normalize the attention scores to probabilities.
        attention_probs0 = attention_scores.softmax(-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout0(attention_probs0)
        # context
        context_layer = torch.matmul(attention_probs, value_layer)
        # --
        if rposi is not None:
            _rval = self.edge_values(rposi)  # [*, Q, K, D]
            _rval2 = torch.matmul(attention_probs.unsqueeze(-2), _rval.unsqueeze(-4)).squeeze(-2)  # [*, H, Q, D]
            context_layer = context_layer + _rval2
        # --
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # output & addnorm
        hidden_states = self.dense(context_layer)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.layer_norm(hidden_states + query)
        return hidden_states  # [*, LQ, D]

class BertFFN(nn.Module):
    def __init__(self, conf: ZBertConf):
        super().__init__()
        config = conf.config
        # --
        # layer1
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = nnutils.ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # layer2
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # --

    def forward(self, hidden_states):
        input0 = hidden_states
        # layer1
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        # layer2
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input0)
        # --
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, conf: ZBertConf):
        super().__init__()
        # --
        self.attention = BertAttention(conf, kv_size=-1, kv_static=False, relC=conf.self_att_relC)  # self-attention
        if conf.add_cross_att:  # cross-attention
            self.crossattention = BertAttention(conf, kv_size=conf.cross_att_size, kv_static=True)
        else:
            self.crossattention = None
        self.ffn = BertFFN(conf)
        # --

    def forward(self, self_t, self_mask_k=None, self_mask_qk=None,
                cross_t=None, cross_mask_k=None, cross_mask_qk=None, cache: Dict = None, rposi=None):
        cache_self, cache_cross = None, None
        if cache is not None:
            if 'self' not in cache:
                cache['self'] = {}
            if 'cross' not in cache:
                cache['cross'] = {}
            cache_self, cache_cross = cache['self'], cache['cross']
        # self-att
        hid_t = self.attention(self_t, self_t, self_t, self_mask_k, self_mask_qk, cache_self, rposi=rposi)  # [*, LQ, D]
        # cross-att
        if self.crossattention is not None:  # [*, LQ, D]
            hid_t = self.crossattention(hid_t, cross_t, cross_t, cross_mask_k, cross_mask_qk, cache_cross)
        # ffn
        hid_t = self.ffn(hid_t)
        return hid_t

class BertEncoder(nn.Module):
    def __init__(self, conf: ZBertConf):
        super().__init__()
        config = conf.config
        # --
        self.layer = nn.ModuleList([BertLayer(conf) for _ in range(config.num_hidden_layers)])
        # --

    def forward(self, self_t, self_mask_k=None, self_mask_qk=None,
                cross_t=None, cross_mask_k=None, cross_mask_qk=None,
                cache: Dict = None, go_layers: List[int] = None, rposi=None):
        all_hidden_states = [self_t]  # [*, L, D]
        cur_hid_t = self_t
        # --
        if go_layers is None:
            go_layers = range(len(self.layer))
        for i in go_layers:
            layer_module = self.layer[i]
            cur_cache = None
            if cache is not None:
                if i not in cache:
                    cache[i] = {}
                cur_cache = cache[i]
            cur_hid_t = layer_module(
                cur_hid_t, self_mask_k, self_mask_qk, cross_t, cross_mask_k, cross_mask_qk, cur_cache, rposi=rposi)
            all_hidden_states.append(cur_hid_t)
        return cur_hid_t, all_hidden_states

class BertModel(nn.Module):
    def __init__(self, conf: ZBertConf):
        super().__init__()
        # --
        self.embeddings = BertEmbeddings(conf)
        self.encoder = BertEncoder(conf)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, extra_embeds=None, self_mask_k=None,
                self_mask_qk=None, cross_t=None, cross_mask_k=None, cross_mask_qk=None,
                cache: Dict = None, go_layers: List[int] = None, rposi=None):
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids, extra_embeds, cache)
        encoder_outputs = self.encoder(
            embedding_output, self_mask_k, self_mask_qk, cross_t, cross_mask_k, cross_mask_qk, cache, go_layers, rposi=rposi)
        return encoder_outputs

# LM output head
class BertLMHead(nn.Module):
    def __init__(self, conf: ZBertConf):
        super().__init__()
        config = conf.config
        # --
        # layer1
        if conf.lmhead_no_hid:
            self.dense = None
            self.layer_norm = None
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            if isinstance(config.hidden_act, str):
                self.transform_act_fn = nnutils.ACT2FN[config.hidden_act]
            else:
                self.transform_act_fn = config.hidden_act
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # layerF
        # self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states):
        # layer1
        if self.dense is not None:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.transform_act_fn(hidden_states)
            hidden_states = self.layer_norm(hidden_states)
        # layerF
        prediction_scores = self.decoder(hidden_states)
        return prediction_scores

# --
# b zgen/model/mods/zbert:?
