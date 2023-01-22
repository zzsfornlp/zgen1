#

# mtl model (in some way a collector)

__all__ = [
    "ZModelConf", "ZModel", "LossHelper",
]

from typing import List, Dict
from collections import OrderedDict, Counter
import torch
from zgen.utils import zlog
from zgen.utils import nn as nnutils
from .node import *
from .task import *
from .med import *

# --

# common format of loss_info is: {'name1': {'sum', 'count', ...}, 'name2': ...}
class LossHelper:
    # compile one loss for one component
    @staticmethod
    def compile_leaf_loss(name: str, loss_sum, loss_count, loss_lambda=1., **other_values):
        if loss_lambda <= 0.:
            return OrderedDict()
        local_dict = {"sum0": loss_sum, "sum": loss_sum*loss_lambda, "count": loss_count, "run": 1}
        local_dict.update(other_values)
        return OrderedDict({name: local_dict})

    # collect all losses for one component
    @staticmethod
    def compile_component_loss(name: str, sub_losses: List[Dict], loss_lambda=1.):
        ret_dict = OrderedDict()
        if loss_lambda <= 0.:
            return ret_dict
        name_prefix = name+"." if name else name
        for one_sub_loss in sub_losses:
            for k, v in one_sub_loss.items():
                ret_dict[f"{name_prefix}{k}"] = v
        for one_item in ret_dict.values():
            one_item["sum"] = one_item["sum"] * loss_lambda
        return ret_dict

    # collect all losses for (possibly) multiple runs
    @staticmethod
    def combine_multiple_losses(inputs: List[Dict]):
        # each input is a flattened loss Dict
        ret = OrderedDict()
        for one_input in inputs:
            if one_input is None:  # skip None
                continue
            for name, leaf_info in one_input.items():
                if name in ret:
                    # adding together
                    target_info = ret[name]
                    for k, v in leaf_info.items():
                        target_info[k] = target_info.get(k, 0.) + v
                else:
                    # direct set
                    ret[name] = leaf_info
        return ret

class ZModelConf(ZNodeConf):
    def __init__(self):
        super().__init__()
        # --
        self.med_conf = ZMediatorConf()
        # share BaseT embeddings
        self.share_baset_embeddings = []  # mod0|mod1
        # --

@node_reg(ZModelConf)
class ZModel(ZNode):
    def __init__(self, conf: ZModelConf):
        super().__init__(conf)
        self.mods: Dict[str, ZMod] = OrderedDict()
        # self.med: ZMediator = None

    def add_mod(self, mod: ZMod):
        _name = mod.name
        assert _name not in self.mods
        self.mods[_name] = mod
        self.add_module(f"M{_name}", mod)
        # --

    def get_mod(self, mname: str, df=None):
        return self.mods.get(mname, df)

    # =====
    # collect all loss
    def collect_loss(self, losses: List[Dict], ret_dict=False):
        final_loss_dict = LossHelper.combine_multiple_losses(losses)  # loss_name -> {}
        if len(final_loss_dict) <= 0:
            return torch.zeros([]), {}  # no loss!
        final_losses = []
        final_losses_dict = OrderedDict()
        ret_info = OrderedDict()
        for loss_name, loss_info in final_loss_dict.items():
            one_real_loss = loss_info['sum']/(loss_info['count']+1e-5)
            # --
            final_losses.append(one_real_loss)  # add scalar-tensor
            final_losses_dict[loss_name] = one_real_loss
            # --
            for k, v in loss_info.items():
                ret_info[f"loss:{loss_name}_{k}"] = float(v.item()) if hasattr(v, "item") else float(v)
            ret_info[f"loss:{loss_name}_div"] = float(one_real_loss.item()) if hasattr(one_real_loss, "item") else float(v)
        if ret_dict:
            return final_losses_dict, ret_info
        else:
            return torch.stack(final_losses).sum() if len(final_losses)>0 else torch.zeros([]), ret_info

    # == load and save models
    # todo(+n): load and save optim states to allow continue training?
    def load(self, path, strict=None):
        if strict is not None:
            nnutils.load_model(self, path, strict=strict)
        else:  # otherwise, first try strict, then relax if there are errors
            try:
                nnutils.load_model(self, path, strict=True)
            except:
                import traceback
                zlog(f"#== Error in strict loading:\n{traceback.format_exc()}\n#==")
                nnutils.load_model(self, path, strict=False)
        zlog(f"Load {self} from {path}.", func="io")

    def save(self, path):
        nnutils.save_model(self, path)
        zlog(f"Save {self} to {path}.", func="io")

    def __repr__(self):
        return f"{self.__class__}(NumParam={self.count_param_number()})"

    def str_details(self):
        return super().__repr__()

    def tie_weights(self):
        for mods in self.conf.share_baset_embeddings:
            name0, name1 = mods.split("|")
            mod0, mod1 = self.mods[name0], self.mods[name1]
            assert mod0.bert.model.embeddings.word_embeddings.weight.shape == mod1.bert.model.embeddings.word_embeddings.weight.shape
            mod0.bert.model.embeddings.word_embeddings.weight = mod1.bert.model.embeddings.word_embeddings.weight
            zlog(f"Tie word embedding weight of {name0} & {name1}!")

    def finish_sr(self):
        # --
        def _tie_weights(module):
            if hasattr(module, 'tie_weights'):
                module.tie_weights()
        # --
        self.apply_pre(_tie_weights)
        zlog(f"Finnish building model: {self}")
        # --

    def forward(self, ibatch, do_loss=False, do_pred=False, **kwargs):
        cur_mods = [self.mods[t] for t in ibatch.dataset.tasks]
        med = ZMediator(self.conf.med_conf)
        med.restart(ibatch)
        # --
        for m in cur_mods:
            m.do_prep(med, **kwargs)
        if do_loss:
            all_losses = []
            info = Counter()
            for m in cur_mods:
                one_loss, one_info = m.do_loss(med, **kwargs)
                all_losses.append(one_loss)
                info += Counter(one_info)
            ret_info = {"inst": len(ibatch), "fb": 1, "fb0": 0}
            ret_info.update(info)
            final_loss, loss_info = self.collect_loss(all_losses)
            ret_info.update(loss_info)
            return final_loss, ret_info
        if do_pred:
            with nnutils.no_grad_env():
                info = Counter()
                for m in cur_mods:
                    one_info = m.do_predict(med, **kwargs)
                    info += Counter(one_info)
                ret_info = {"inst": len(ibatch), "ff": 1}
                ret_info.update(info)
            return ret_info
        # --

# --
# b zgen/model/core/model:138
