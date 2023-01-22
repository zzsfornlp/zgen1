#

# specific model
# not an elegant solution, but ...

__all__ = [
    "IncrEmbedConf", "IncrEmbed", "IncrModelConf", "IncrModel",
]

import torch
from zgen.utils import ConfEntryChoices, zlog, ZObject
from zgen.utils import nn as nnutils
from ...core import *
from ...mods import BertAttention

# --
# special position embeddings
class IncrEmbedConf(ZNodeConf):
    def __init__(self):
        self.esize = -1  # model size
        # --
        self.comb_method = ConfEntryChoices({"avg": None, "aff": ZAffineConf.direct_conf(act='elu')}, "aff")
        # --

@node_reg(IncrEmbedConf)
class IncrEmbed(ZNode):
    def __init__(self, conf: IncrEmbedConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IncrEmbedConf = self.conf
        # --
        _dim = conf.esize
        self.special = torch.nn.Embedding(2, _dim)
        if conf.comb_method is None:  # simply average
            self.comb_f = lambda x,y: (x+y)/2
        elif isinstance(conf.comb_method, ZAffineConf):
            self.comb_node = ZAffineNode(conf.comb_method, isize=[_dim,_dim], osize=_dim)
            self.comb_f = lambda x,y: self.comb_node([x,y])
        else:
            raise NotImplementedError()
        # --
        self.to(nnutils.DEFAULT_DEVICE)
        # --

    def forward_special(self, idxes_t):
        ret = self.special(idxes_t)  # [*, D]
        return ret

    # [*, D]x2
    def forward(self, left_es, right_es):
        ret = self.comb_f(left_es, right_es)
        return ret  # [*, D]

# --
# IncrModel

# model0: forward-all = incr_embed=False,incr_grow_hid=False,incr_layers=0
# model1: caching = incr_embed=True,incr_grow_hid=False,incr_layers=??
# model2: growing = incr_embed=True,incr_grow_hid=True,incr_layers=??
class IncrModelConf(ZNodeConf):
    def __init__(self):
        # use special embeddings for incr?
        self.incr_embed = ConfEntryChoices({"no": None, "yes": IncrEmbedConf()}, "no")
        # incremental building hidden layers
        self.incr_grow_hid = False
        # how to split layers
        self.incr_layers = 0  # all_layers = incr_layers + score_layers
        # use incr relative?
        self.incr_rel = False
        # --
        # extra feature combiner
        self.comb = ZAffineConf.direct_conf(act='elu')
        # extraly attending to last_stage
        self.last_stage_atthead = 0  # 0 means nope
        # --

@node_reg(IncrModelConf)
class IncrModel(ZNode):
    def __init__(self, conf: IncrModelConf, mod, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IncrModelConf = self.conf
        # check and setup mod
        self.incr_rel = conf.incr_rel  # use relative!
        assert mod.bert.lmhead is not None
        if conf.incr_embed is None:  # model0
            zlog("Choose model0: forward-all")
            assert (not conf.incr_grow_hid) and (conf.incr_layers == 0)
        elif not conf.incr_grow_hid:  # model1
            zlog("Choose model1: caching")
            assert not mod.bert.has_posi_embeds, "Should not have ABS posi_embeds"
        else:
            zlog("Choose model2: growing")
            assert not mod.bert.has_posi_embeds, "Should not have ABS posi_embeds"
        self.setattr_borrow('mod', mod)  # self.mod: ZDecoderIlm = mod
        # check layers
        _incr_layers = conf.incr_layers
        _all_layers = mod.bert.num_layers
        assert _incr_layers <= _all_layers and _incr_layers >= 0
        self.incr_layers, self.score_layers = list(range(_incr_layers)), list(range(_incr_layers, _all_layers))
        zlog(f"Separate the base model as {self.incr_layers} & {self.score_layers}")
        # --
        # incr-emb
        if conf.incr_embed is not None:
            self.incr_emb = conf.incr_embed.make_node(esize=mod.bert.hidden_size)
        else:
            self.incr_emb = None
        # --
        # extra feature combiner
        _hid_dim = mod.bert.hidden_size
        self.comb = ZAffineNode(conf.comb, isize=[_hid_dim, _hid_dim], osize=_hid_dim)
        # --
        # extra (last-stage) attention
        if conf.last_stage_atthead > 0:
            # note: fake a config
            _conf = ZObject(
                config=ZObject(hidden_size=_hid_dim, num_attention_heads=conf.last_stage_atthead,
                               attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1, layer_norm_eps=1e-12))
            self.last_att = BertAttention(_conf).to(nnutils.DEFAULT_DEVICE)
        else:
            self.last_att = None
        # --

    @property
    def is_model0(self):  # model0: forward-all
        return self.incr_emb is None

    # run incremental building at the very start: simply forward [cls, sep]
    def run_incr_init(self, bsize, cache, skip_forw: bool, **forw_kwargs):
        _shape = [bsize, 2]  # [*, 2]
        # --
        # posi_t
        if self.is_model0 or self.incr_rel:
            posi_t = None
        else:
            _tmp_pidx = nnutils.zeros(_shape, dtype=nnutils.DEFAULT_INT)  # [*, 2]
            _tmp_pidx[..., 1] = 1  # [*, [0,1]]
            posi_t = self.incr_emb.forward_special(_tmp_pidx)  # [*, 2, D]
        # then forward things?
        if skip_forw or self.is_model0:  # no incr in model0!
            ret_hid_t, ret_mask_t = None, None
        else:
            svoc = self.mod.ztask.special_vocab
            bos_id, eos_id = svoc.cls_token_id, svoc.sep_token_id
            id_t = nnutils.constants(_shape, value=bos_id, dtype=nnutils.DEFAULT_INT)  # [*, 2]
            id_t[..., 1] = eos_id
            mask_t = nnutils.constants(_shape, value=1., dtype=nnutils.DEFAULT_FLOAT)
            # --
            if self.incr_rel:  # note: no need from outside!
                rposi = nnutils.input_idx([[[0, 1], [-1, 0]]] * bsize)  # [*, 2, 2]
            else:
                rposi = None
            # --
            ret_hid_t, ret_mask_t = self._forw_incr(id_t, mask_t, posi_t, None, None, cache, rposi=rposi, **forw_kwargs)
        return posi_t, ret_hid_t, ret_mask_t

    # run incremental building (with incr-layers): [*, ??, ?]
    def run_incr(self, cur_id_t, cur_mask_t, cur_left_t, cur_right_t,
                 prev_posi_t, prev_mask_t, prev_hid_t, cache, skip_forw: bool, **forw_kwargs):
        # first obtain posi_t
        if self.is_model0 or self.incr_rel:
            cur_posi_t = None
            ret_posi_t = None
        else:
            _arange_t = nnutils.get_arange_t(cur_id_t.shape[0], 1)  # [*, 1]
            left_es, right_es = prev_posi_t[_arange_t, cur_left_t], prev_posi_t[_arange_t, cur_right_t]  # [*, kk, D]
            cur_posi_t = self.incr_emb.forward(left_es, right_es)  # [*, kk, D]
            ret_posi_t = torch.cat([prev_posi_t, cur_posi_t], -2)  # [*, prev+kk, D]
        # then forward things?
        if skip_forw or self.is_model0:  # no incr in model0!
            ret_hid_t, ret_mask_t = None, None
        else:
            ret_hid_t, ret_mask_t = self._forw_incr(cur_id_t, cur_mask_t, cur_posi_t, prev_mask_t, prev_hid_t, cache, **forw_kwargs)
        return ret_posi_t, ret_hid_t, ret_mask_t

    # forward for incr part: [*, kk]; [*, prev]
    def _forw_incr(self, cur_id_t, cur_mask_t, cur_posi_t, prev_mask_t, prev_hid_t, cache, **forw_kwargs):
        conf: IncrModelConf = self.conf
        bmod = self.mod.bert.model
        # --
        # get new embeddings
        cur_emb_t = bmod.embeddings(cur_id_t, extra_embeds=cur_posi_t, cache=cache)  # [*, kk, D]
        new_mask_t = cur_mask_t if prev_mask_t is None else torch.cat([prev_mask_t, cur_mask_t], -1)  # [*, A]
        # get new hid
        if conf.incr_grow_hid:  # growing
            new_input_t = cur_emb_t if prev_hid_t is None else torch.cat([prev_hid_t, cur_emb_t], -2)  # [*, A, D]
            new_hid_t, _ = bmod.encoder(
                new_input_t, self_mask_k=new_mask_t, cache=cache, go_layers=self.incr_layers, **forw_kwargs)  # [*, A, D]
            # clear cache of self-atts!!
            for ii in self.incr_layers:
                cache[ii]['self'].clear()
        else:  # caching
            cur_hid_t, _ = bmod.encoder(
                cur_emb_t, self_mask_k=cur_mask_t, cache=cache, go_layers=self.incr_layers, **forw_kwargs)  # [*, k, D]
            new_hid_t = cur_hid_t if prev_hid_t is None else torch.cat([prev_hid_t, cur_hid_t], -2)  # [*, A, D]
        return new_hid_t, new_mask_t  # [*, A, D]

    # run scoring feat (with score-layers)
    # 3x[*, rlen], 2x[*, clen], 2x[*, ??]
    def run_score_feat(self, hid_t, mask_t, last_stage_t, input_id_t, input_mask_t, left_t, right_t, cache, **forw_kwargs):
        bmod = self.mod.bert.model
        # --
        if self.is_model0:  # model0, forward from scratch!
            posi_idx_t = input_mask_t.to(nnutils.DEFAULT_INT).cumsum(-1) - 1  # [*, clen]
            hid_t = bmod.embeddings(input_id_t, position_ids=posi_idx_t)  # todo(note): no putting cache here!!!
            mask_t = input_mask_t
        # go score: # [*, rlen]
        out_hid_t, _ = bmod.encoder(hid_t, self_mask_k=mask_t, cache=cache, go_layers=self.score_layers, **forw_kwargs)
        # clear cache of self-atts!
        for ii in self.score_layers:
            cache[ii]['self'].clear()
        # gather and calculate
        _arange_t = nnutils.get_arange_t(left_t.shape[0], 1)  # [*, 1]
        hid_t0, hid_t1 = out_hid_t[_arange_t, left_t], out_hid_t[_arange_t, right_t]  # [*, ??, D]
        feat_t = self.comb([hid_t0, hid_t1])  # [*, ??, D]
        # last att
        if self.last_att is not None:
            feat_t = self.last_att(feat_t, hid_t, hid_t, mask_k=last_stage_t)  # [*, ??, D]
        # --
        return feat_t

    def run_score_head(self, feat_t):
        score_t = self.mod.bert.lmhead(feat_t)
        return score_t

    # run with canvas for predictions: only include [from, to]
    def run_canvas(self, canvas, from_stage: int, to_stage: int, detach_growing: bool, **forw_kwargs):
        bmod = self.mod.bert.model
        conf: IncrModelConf = self.conf
        # --
        canvas_stage_t = canvas.stage_t.clone()  # [*, L]
        canvas_stage_t[canvas_stage_t<0] = canvas.cur_stage + 100  # make a large one for convenience
        run_cache = canvas.cache.run_cache
        left_ts, right_ts = canvas.history.left_ts, canvas.history.right_ts
        # --
        ret_feats = []
        if self.is_model0:  # model0: re-forward every time
            for cur_stage in range(from_stage, to_stage+1):
                _cur_mask_t = (canvas_stage_t < cur_stage).to(nnutils.DEFAULT_FLOAT)  # [*, L]
                feat_t = self.run_score_feat(
                    None, None, None, canvas.id_t, _cur_mask_t, left_ts[cur_stage-1], right_ts[cur_stage-1],
                    run_cache, **forw_kwargs)  # [*, ??, D]
                ret_feats.append(feat_t)
        else:
            # prepare embed
            posi_t = canvas.cache.posi_t
            _arange_t = nnutils.get_arange_t(canvas.id_t.shape[0], 1)  # [*, 1]
            if posi_t is None:
                nat_posi_t = None
            else:
                nat_posi_t = posi_t[_arange_t, canvas.cache.imap_t]  # [*, L, D]
            emb_t = bmod.embeddings(canvas.id_t, extra_embeds=nat_posi_t, cache=None)  # [*, L, D]
            if not conf.incr_grow_hid:  # model1: caching
                if self.incr_rel:
                    from .help_search import get_rposi_full
                    rposi = get_rposi_full(canvas_stage_t)
                else:
                    rposi = None
                # [*, L, L], can only look at >=stage when incr!
                causal_mask_t = (canvas_stage_t.unsqueeze(-1) >= canvas_stage_t.unsqueeze(-2)).to(nnutils.DEFAULT_FLOAT)
                hid_t, _ = bmod.encoder(emb_t, self_mask_k=canvas.mask_t, self_mask_qk=causal_mask_t,
                                        cache=run_cache, go_layers=self.incr_layers, rposi=rposi, **forw_kwargs)  # [*, L, D]
                # then collect all the feats
                for cur_stage in range(from_stage, to_stage+1):
                    _cur_mask_t = (canvas_stage_t < cur_stage).to(nnutils.DEFAULT_FLOAT)  # [*, L]
                    _stage_m1 = cur_stage - 1
                    _last_stage_t = (canvas_stage_t==_stage_m1).to(nnutils.DEFAULT_FLOAT)  # [*, L]
                    feat_t = self.run_score_feat(
                        hid_t, _cur_mask_t, _last_stage_t, None, None, left_ts[_stage_m1], right_ts[_stage_m1],
                        run_cache, **forw_kwargs)  # [*, ??, D]
                    ret_feats.append(feat_t)
            else:  # model2: growing
                hid_t = emb_t  # [*, L, D]
                for cur_stage in range(1, to_stage+1):
                    # forward incr
                    _cur_mask_t = (canvas_stage_t < cur_stage).to(nnutils.DEFAULT_FLOAT)  # [*, L]
                    new_hid_t, _ = bmod.encoder(
                        hid_t, self_mask_k=_cur_mask_t, cache=run_cache, go_layers=self.incr_layers, **forw_kwargs)  # [*, L, D]
                    if detach_growing and cur_stage < from_stage:  # save mem
                        new_hid_t = new_hid_t.detach()
                    # update: new = new * mask + emb * (1-mask) = emb - mask * (new-emb)
                    hid_t = emb_t + _cur_mask_t.unsqueeze(-1) * (new_hid_t - emb_t)  # [*, L, D]
                    # clear cache of self-atts!!
                    for ii in self.incr_layers:
                        run_cache[ii]['self'].clear()
                    # get feat
                    if cur_stage >= from_stage:
                        _stage_m1 = cur_stage - 1
                        _last_stage_t = (canvas_stage_t==_stage_m1).to(nnutils.DEFAULT_FLOAT)  # [*, L]
                        feat_t = self.run_score_feat(
                            hid_t, _cur_mask_t, _last_stage_t, None, None, left_ts[_stage_m1], right_ts[_stage_m1],
                            run_cache, **forw_kwargs)  # [*, ??, D]
                        ret_feats.append(feat_t)
        # --
        # breakpoint()
        # note: debug with: ( ((canvas.cache._tmp_hid_t[_arange_t, canvas.cache.imap_t.clamp(min=0)] - hid_t).float().abs() * canvas.mask_t.unsqueeze(-1).float()) > 1e-5).sum(-1)
        return ret_feats  # *[*, ??, D]

# --
# b zgen/model/tasks/dec_ilm/help_incr:?
