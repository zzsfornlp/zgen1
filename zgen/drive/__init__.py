#

# some common functions for running, go!

from zgen.utils import Conf, ConfEntryChoices
from .data_center import *
from .task_center import *
from .run_center import *
from ..model.model0 import ZModelConf0

class ZOverallConf(Conf):
    def __init__(self):
        self.tconf = TaskCenterConf()  # task conf
        self.dconf = DataCenterConf()  # data conf
        self.rconf = RunCenterConf()  # run conf
        self.mconf = ConfEntryChoices({"m0": ZModelConf0()}, "m0")
        # --
        self.extra_model_specs = []
        # --

# --
extra_models = []
# --
# note: call this only once!!
def load_extra_models(extra_model_specs):
    from zgen.utils import init_everything, zlog
    from zgen.utils import nn as nnutils
    for one_spec in extra_model_specs:
        one_conf_file, one_model_file, one_vocab_dir = one_spec.split(";;")
        zlog(f"# ==\nStarting loading extra_model with {one_spec}")
        conf: ZOverallConf = init_everything(
            ZOverallConf(), [one_conf_file], add_utils=False, add_nn=False, do_check=False, add_global_key='')
        t_center = TaskCenter(conf.tconf)
        t_center.load_vocabs(one_vocab_dir)
        model = conf.mconf.make_node()
        t_center.build_mods(model)
        model.finish_sr()  # note: build sr before possible loading in testing!!
        model.load(one_model_file)
        model.to(nnutils.DEFAULT_FLOAT)  # move to default float!!
        model.eval()  # by default eval mode!!
        # --
        zlog(f"# ==\nFinished loading extra_model[#{len(extra_models)}]: {model}")
        extra_models.append(model)
        # --

# --
# b zgen/drive/__init__:30
