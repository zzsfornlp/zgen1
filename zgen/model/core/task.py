#

# the base ones for ZTask
# todo(+N): maybe the current design is not good, but ZTask and ZMod are indeed tightly coupled together ...

__all__ = [
    "ZTaskConf", "ZTask", "ZModConf", "ZMod",
]

# --

import os
from typing import List
from zgen.utils import Conf, zlog, default_pickle_serializer
from .node import *
from .med import *

# --
# ztask

class ZTaskConf(Conf):
    def __init__(self):
        self.name = ""
        self.eval_weight = 1.  # weight for final eval

    def build_task(self):
        raise NotImplementedError()

class ZTask:
    def __init__(self, conf: ZTaskConf):
        self.conf = conf
        self.name = conf.name
        # --
        self.vpack = None  # to be built or load
        self.mod = None  # to be built

    def __repr__(self):
        return f"ZTask({self.name})"

    # --
    # part 1: data related

    # -- part 1.1: vocab
    # build vocabs with the datasets
    def build_vocab(self, datasets: List):
        raise NotImplementedError()

    def save_vocab(self, v_dir: str):
        vp_file = os.path.join(v_dir, f"v_{self.name}.pkl")
        if self.vpack is not None:
            default_pickle_serializer.to_file(self.vpack, vp_file)
            zlog(f"Save vocabs ``{self.vpack}'' for {self} to {vp_file}")

    def load_vocab(self, v_dir: str):  # return whether succeed!
        vp_file = os.path.join(v_dir, f"v_{self.name}.pkl")
        if os.path.exists(vp_file):
            self.vpack = default_pickle_serializer.from_file(vp_file)
            zlog(f"Load vocabs ``{self.vpack}'' for {self} from {vp_file}")
            return True
        else:
            self.vpack = None  # not_found
            return False
        # --

    # -- part 1.2: data
    # prepare function for dataset stream!
    def prep_f(self, inst, dataset) -> object:
        return inst

    # -- part 1.3: eval
    # eval and return metric (by default nope)
    def eval_insts(self, gold_insts, pred_insts, quite=False):
        return None

    # --
    # part 2: model related
    def build_mod(self, model):
        raise NotImplementedError()

# --
# zmod

class ZModConf(ZNodeConf):
    def __init__(self):
        super().__init__()
        # --

@node_reg(ZModConf)
class ZMod(ZNode):
    def __init__(self, conf: ZModConf, ztask: ZTask, **kwargs):
        super().__init__(conf, **kwargs)
        self.ztask = ztask
        # --

    @property
    def name(self):
        return self.ztask.name

    # ==
    # to be implemented

    def do_prep(self, med: ZMediator, *args, **kwargs):  # sth to prepare before enc?
        pass  # by default nothing!

    def do_loss(self, med: ZMediator, *args, **kwargs):
        raise NotImplementedError()

    def do_predict(self, med: ZMediator, *args, **kwargs):
        return self.do_loss(med, *args, **kwargs)[1]

    def do_score(self, med: ZMediator, *args, **kwargs):
        return self.do_predict(med, *args, **kwargs)
