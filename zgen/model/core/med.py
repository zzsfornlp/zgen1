#

# mediator (actually temp value storer)

__all__ = [
    "ZMediatorConf", "ZMediator", "ZValue",
]

import torch
from typing import List, Dict
from collections import OrderedDict, Counter
from zgen.utils import Conf, Constants

# --
# mediate between encoder and decoders
class ZMediatorConf(Conf):
    def __init__(self):
        pass

class ZMediator:
    def __init__(self, conf: ZMediatorConf = None):
        self.conf = ZMediatorConf.direct_conf(conf)
        conf: ZMediatorConf = self.conf
        # --
        # self.mods = mods
        # state
        self.ibatch = None
        self.cache = {}

    # new batch
    def restart(self, ibatch):
        self.ibatch = ibatch
        self.cache.clear()  # simply clean them

    # --
    # cached values

    def set_cache(self, k, v, app=False, app_info=None, assert_nonexist=True):
        _cc = self.cache
        # --
        if app:  # appending mode
            zv = _cc.get(k)
            if zv is None:
                zv = ZValue()
                _cc[k] = zv
            zv.append(v, app_info)
        else:  # adding like a dict
            if assert_nonexist:
                assert k not in _cc
            _cc[k] = v
        # --

    def get_cache(self, k, df=None):
        return self.cache.get(k, df)

    def get_cache_val(self, k, **kwargs):
        val = self.get_cache(k)
        return val.get_val(**kwargs)

# (layered/multiple) value container: can store hids, attns, scores, ...
# -> note: basically a list, the values should have the same shape!!
class ZValue:
    def __init__(self):
        self.vals = []  # List[val]
        self.infos = []  # List[info]
        self.vmap = OrderedDict()  # info->val
        # --
        self._cache = OrderedDict()  # cached value, for example, selected ones!

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, item):
        return self.vals[item]

    def append(self, v, info=None):
        if v is not None:  # note: ignore none!!
            self.vals.append(v)
            self.infos.append(info)
            if info is not None:  # extra store!
                assert info not in self.vmap
                self.vmap[info] = v
            # clear the cache whenever we add new things!
            self._cache.clear()

    # get val (if idx is None, then stack all!!)
    def get_val(self, idx=-1, stack_dim=-2, signature=None, no_cache=False):
        _k = (idx, stack_dim, signature)  # key for cache
        ret = None
        if not no_cache:
            ret = self._cache.get(_k)
        if ret is None:  # calculate!!
            if idx is None:
                v0 = torch.stack(self.vals, dim=stack_dim)  # [*, ilen, ND, *]
            else:
                v0 = self.vals[idx]  # [*, ilen, *]
            ret = v0
            if not no_cache:
                self._cache[_k] = ret   # store cache
        # --
        return ret
