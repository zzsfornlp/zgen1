#

# collection of tasks

__all__ = [
    "TaskCenterConf", "TaskCenter",
]

from typing import List
from collections import OrderedDict
from zgen.utils import Conf, ConfEntryChoices, zlog, zglob1
from .data_center import DataCenter
from ..model import ZTask, ZTaskConf, ZTaskEncBertConf, ZTaskMlmConf, ZTaskInslmConf, ZTaskSlmConf, ZTaskIlmConf

# --
class TaskCenterConf(Conf):
    def __init__(self):
        # vocab dir
        self.vocab_save_dir = "./"  # place to store all vocabs
        self.vocab_load_dir = "./"  # place to load pre-built vocabs
        self.vocab_force_rebuild = False  # force rebuild all vocabs
        # --
        self.enc = ConfEntryChoices({"bert": ZTaskEncBertConf(), "no": None}, "no")
        self.mlm = ConfEntryChoices({"yes": ZTaskMlmConf(), "no": None}, "no")
        self.inslm = ConfEntryChoices({"yes": ZTaskInslmConf(), "no": None}, "no")
        self.slm = ConfEntryChoices({"yes": ZTaskSlmConf(), "no": None}, "no")
        self.ilm = ConfEntryChoices({"yes": ZTaskIlmConf(), "no": None}, "no")
        # --

    @classmethod
    def _get_type_hints(cls):
        return {"vocab_load_dir": "zglob1"}  # easier finding!

    def get_all_tconfs(self):
        names = ["enc", "mlm", "inslm", "slm", "ilm"]
        ret = [getattr(self, n) for n in names]
        ret = [z for z in ret if isinstance(z, ZTaskConf)]
        return ret

class TaskCenter:
    def __init__(self, conf: TaskCenterConf):
        self.conf = conf
        # --
        # build them
        self.tasks = OrderedDict()
        for tconf in conf.get_all_tconfs():
            task: ZTask = tconf.build_task()
            assert task.name not in self.tasks, "Repeated task!!"
            self.tasks[task.name] = task
        # --
        zlog(f"Build TaskCenter ok: {self}")

    def __repr__(self):
        return f"TaskCenter with: {list(self.tasks.keys())}"

    def build_vocabs(self, d_center: DataCenter):
        load_vdir = self.conf.vocab_load_dir
        save_vdir = self.conf.vocab_save_dir
        # --
        # first try load vocabs
        if not self.conf.vocab_force_rebuild:
            load_info = self.load_vocabs(load_vdir, quiet=True)
        else:
            load_info = OrderedDict()
        load_names = [k for k,v in load_info.items() if v]
        # then build for those not loaded!
        build_names = []
        for n, t in self.tasks.items():
            if not load_info.get(n, False):  # if not loaded!
                t_datasets = d_center.get_datasets(task=n)  # obtain by task name!!
                # --
                assert t.vpack is None
                t.vpack = t.build_vocab(t_datasets)
                # --
                build_names.append(n)
                if save_vdir is not None:
                    t.save_vocab(save_vdir)
        zlog(f"Build vocabs: load {load_names} from {load_vdir}, build {build_names}")
        # --

    def load_vocabs(self, v_dir: str, quiet=False):
        info = OrderedDict()
        for n, t in self.tasks.items():
            info[n] = t.load_vocab(v_dir)
        if not quiet:
            zlog(f"Load vocabs from {v_dir}, success={info}")
        return info
        # --

    def save_vocabs(self, v_dir: str):
        for n, t in self.tasks.items():
            t.save_vocab(v_dir)
        zlog(f"Save vocabs to {v_dir}")
        # --

    def add_preps(self, d_center: DataCenter):
        # put preps into the pipe of dataset
        for dataset in d_center.get_datasets():
            ffs = [self.tasks[z].prep_f for z in dataset.tasks]
            dataset.add_preprocessors(ffs)
        # --

    def build_mods(self, model):
        for t in self.tasks.values():
            assert t.mod is None
            t.mod = t.build_mod(model)  # "build_mod" only needs to return a built one!!
            model.add_mod(t.mod)
        # --

# --
# b zgen/drive/task_center:122
