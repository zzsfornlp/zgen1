#

# collection of datasets

__all__ = [
    "DataCenterConf", "DataCenter",
]

from collections import OrderedDict
from typing import List
from copy import deepcopy
import numpy as np
from zgen.utils import Conf, ConfEntryChoices, zlog, Random, ScheduledValue
from zgen.data import ZDatasetConf, ZDataset

# --
_DATA_MAXN=10  # this should be enough!!
class DataCenterConf(Conf):
    def __init__(self):
        # could be multiple ones!!
        for ii in range(_DATA_MAXN):
            for wset in ["train", "dev", "test"]:
                setattr(self, f"{wset}{ii}", ZDatasetConf())
        self.testM = ZDatasetConf()  # shortcut: main-test that can override others!
        # --

    def get_all_dconfs(self, wset: str):
        ret = [getattr(self, f"{wset}{ii}") for ii in range(_DATA_MAXN)]
        ret = [z for z in ret if z.has_files()]  # ignore empty ones!
        return ret

class DataCenter:
    def __init__(self, conf: DataCenterConf, specified_wset=None):
        self.conf = conf
        # --
        # load and prepare them
        self.datasets = OrderedDict()  # note: two-layered naming! (wset, data_name)
        self.train_sample_svs = OrderedDict()  # sv:sample_rate for training
        if specified_wset is None:
            specified_wset = ["train", "dev", "test"]  # by default read all!
        for wset in specified_wset:
            trg = OrderedDict()
            if wset == "test" and conf.testM.has_files():
                dconfs = [conf.testM]
            else:
                dconfs = conf.get_all_dconfs(wset)
            for ii, dconf in enumerate(dconfs):
                if not dconf.data_name:
                    dconf.data_name = f"{wset}{ii}"
                d = ZDataset(dconf, wset)
                assert d.name not in trg
                trg[d.name] = d
                # --
                if wset == "train":
                    self.train_sample_svs[d.name] = ScheduledValue(f"sr_{d.name}", dconf.data_sample_rate)
            self.datasets[wset] = trg
        # --
        zlog(f"Build DataCenter ok: {self}")

    def __repr__(self):
        ss = [f"{k}:{len(v)}" for k,v in self.datasets.items()]
        return f"DataCenter({','.join(ss)})"

    def get_scheduled_values(self):
        return OrderedDict([(f"_sr_{k}",v) for k,v in self.train_sample_svs.items()])

    def get_datasets(self, wset=None, dname=None, task=None, extra_filter=None):
        # filter by wset
        if wset is None:
            cand_groups: List[OrderedDict] = list(self.datasets.values())
        else:
            cand_groups: List[OrderedDict] = [self.datasets[wset]]
        # filter by group
        if dname is None:
            candidates: List[ZDataset] = [z for g in cand_groups for z in g.values()]
        else:
            candidates: List[ZDataset] = [g[dname] for g in cand_groups]
        # filter by task
        if task is not None:
            candidates = [z for z in candidates if (task in z.tasks)]
        # filter by filter
        if extra_filter is not None:
            candidates = [z for z in candidates if extra_filter(z)]
        return candidates

    # yield yielder to further yield batches (loop forever)
    def yield_train_yielder(self):
        all_yielders = []
        all_svs = []
        for data_name, data_set in self.datasets["train"].items():
            all_yielders.append(data_set.yield_batches())
            all_svs.append(self.train_sample_svs[data_name])
        _gen = Random.get_generator('stream')
        _n_groups = len(all_svs)
        while True:
            if len(all_svs) == 1:
                cur_gidx = 0  # simply 1
            else:
                pvals = np.asarray([z.value for z in all_svs])
                pvals = pvals / pvals.sum()
                cur_gidx = _gen.choice(_n_groups, p=pvals)  # choose group
            # choose that one!
            chosen_yielder = all_yielders[cur_gidx]
            yield chosen_yielder
        # --

# --
# b zgen/drive/data_center:103
