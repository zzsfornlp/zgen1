#

# some basic components

__all__ = [
    "ZNodeConf", "node_reg", "ZNode", "ActivationHelper", "ZAffineConf", "ZAffineNode",
]

from typing import Type, Dict
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from zgen.utils import Conf, zlog, get_class_id, DictHelper
from zgen.utils import nn as nnutils

# --
# node in a model
class ZNodeConf(Conf):
    _CONF_MAP = {}
    _NODE_MAP = {}

    @staticmethod
    def get_conf_type(cls: Type, df):
        _k = get_class_id(cls, use_mod=True)
        return ZNodeConf._CONF_MAP.get(_k, df)

    @staticmethod
    def get_node_type(cls: Type, df):
        _k = get_class_id(cls, use_mod=True)
        return ZNodeConf._NODE_MAP.get(_k, df)

    def make_node(self, *args, **kwargs):
        cls = ZNodeConf.get_node_type(self.__class__, None)
        return cls(self, *args, **kwargs)

# reg for both conf and node
def node_reg(conf_type: Type):
    def _f(cls: Type):
        # reg for conf
        _m = ZNodeConf._CONF_MAP
        _k = get_class_id(cls, use_mod=True)
        assert _k not in _m
        _m[_k] = conf_type
        # reg for node
        _m2 = ZNodeConf._NODE_MAP
        _k2 = get_class_id(conf_type, use_mod=True)
        if _k2 not in _m2:  # note: reg the first one!!
            _m2[_k2] = cls
        # --
        return cls
    return _f

@node_reg(ZNodeConf)
class ZNode(torch.nn.Module):
    def __init__(self, conf: ZNodeConf, **kwargs):
        super().__init__()
        self.conf = self.setup_conf(conf, **kwargs)

    # count number of parameters
    def count_param_number(self, recurse=True):
        count = 0
        list_params = self.parameters(recurse=recurse)
        for p in list_params:
            count += np.prod(p.shape)
        return int(count)

    def setup_conf(self, conf: ZNodeConf, **kwargs):
        if conf is None:
            conf_type = ZNodeConf.get_conf_type(self.__class__, ZNodeConf)  # by default the basic one!
            conf = conf_type()
        else:  # todo(note): always copy to local!
            conf = conf.copy()
        conf.direct_update(_assert_exists=True, **kwargs)
        # conf._do_validate()
        conf.validate()  # note: here we do full validate to setup the private conf!
        return conf

    def reset_params_and_modules(self):
        # first reset current params
        self.reset_parameters()
        # then for all sub-modules
        for m in self.children():
            if isinstance(m, ZNode):
                m.reset_params_and_modules()
            elif hasattr(m, 'reset_parameters'):  # some Modules have this function!
                m.reset_parameters()
            # todo(note): ignore others

    # do not go through Model.__setattr__, thus will not add it!
    def setattr_borrow(self, key: str, value: object, assert_nonexist=True):
        if assert_nonexist:
            assert not hasattr(self, key)
        object.__setattr__(self, key, value)

    # the overall one
    def get_scheduled_values(self) -> Dict:
        ret = OrderedDict()
        for cname, m in self.named_children():
            if isinstance(m, ZNode):
                DictHelper.update_dict(ret, m.get_scheduled_values(), key_prefix=cname+".")
        DictHelper.update_dict(ret, self._get_scheduled_values(), key_prefix="")
        return ret

    # --
    # note: to be implemented, by default nothing, only for special ones
    def _get_scheduled_values(self) -> Dict:
        return OrderedDict()

    # todo(note): only reset for the current layer
    def reset_parameters(self):
        pass  # by default no params to reset!

    # todo(note): override for pretty printing
    def extra_repr(self) -> str:
        return f"{self.__class__.__name__}"

    def tie_weights(self):
        pass

    def apply_pre(self, fn):  # first self
        fn(self)
        for module in self.children():
            module.apply(fn)
        return self

# --
# some useful ones

class ActivationHelper(object):
    ACTIVATIONS = {"tanh": torch.tanh, "relu": torch.relu, "elu": F.elu, "gelu": F.gelu,
                   "sigmoid": torch.sigmoid, "linear": (lambda x:x), "softmax": (lambda x,d=-1: x.softmax(d))}
    # reduction for seq after conv
    POOLINGS = {"max": (lambda x,d: torch.max(x, d)[0]), "min": (lambda x,d: torch.min(x, d)[0]),
                "avg": (lambda x,d: torch.mean(x, d)), "none": (lambda x,d: x)}

    @staticmethod
    def get_act(name):
        return ActivationHelper.ACTIVATIONS[name]

    @staticmethod
    def get_pool(name):
        return ActivationHelper.POOLINGS[name]

class ZAffineConf(ZNodeConf):
    def __init__(self):
        self.isize = []
        self.osize = 512
        self.act = 'linear'
        self.drop_rate = 0.1
        # --

@node_reg(ZAffineConf)
class ZAffineNode(ZNode):
    def __init__(self, conf: ZAffineConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZAffineConf = self.conf
        # --
        self.linear = torch.nn.Linear(sum(int(z) for z in conf.isize), conf.osize)
        self.act = ActivationHelper.get_act(conf.act)
        self.drop = torch.nn.Dropout(conf.drop_rate)
        # --
        self.to(nnutils.DEFAULT_DEVICE)
        # --

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            input_t = torch.cat(inputs, -1)
        else:
            input_t = inputs  # [In]
        hid_t = self.linear(input_t)  # [Out]
        hid_t = self.act(hid_t)
        hid_t = self.drop(hid_t)
        return hid_t
