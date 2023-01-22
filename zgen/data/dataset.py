#

# dataset, with which to yield data

__all__ = ["ZDatasetConf", "ZDataset", "ZDataPreprocessor"]

import os
import re
from typing import List, Callable
from collections import OrderedDict
from zgen.utils import Conf, zlog, Random, Registrable, Timer, Constants, ConfEntryChoices, SVConf
from .rw import MultiFileReaderConf, MultiFileReader, JsonWriterConf, JsonWriter
from .stream import BatchArranger, IterStreamer, WrapperStreamer, CacheStreamerConf, CacheStreamer
from .inst import DataBatch

# --
# for example, @ZDataPreprocessor.reg_decorator("??") def _prep_??(inst): ...
class ZDataPreprocessor(Registrable):
    def __call__(self, inst, dataset):
        return inst

@ZDataPreprocessor.reg_decorator('fake_idxes')
def _prep_fake_idxes(inst, dataset):  # a fake one for early debugging
    inst.idxes = [1] * len(inst.tokens)
    return inst
# --

# --
class ZDataBatcherConf(Conf):
    def __init__(self):
        # general
        self.batch_size = 1024
        self.batch_size_f = 'src_len'  # or "lambda x: 1" / "lambda x: len(x)*len(x.events)"
        self.batch_maxi_bsize = 50
        self.sort_size_f = 'src_len'
        # simple len constraint
        self.filter_dump_f = 'src_len'
        self.filter_min_length = 0
        self.filter_max_length = Constants.INT_PRAC_MAX
        # shuffle buckets
        self.bucket_shuffle_times = 1

class ZDataBatcher:
    # --
    _LEN_F = {
        '1': (lambda x: 1),
        'len': (lambda x: len(x)), 'both_len': (lambda x: x.src_len+x.trg_len),
        'src_len': (lambda x: x.src_len), 'trg_len': (lambda x: x.trg_len),
        'max_len': (lambda x: max(x.src_len, x.trg_len)),
    }
    _DUMP_F = {
        'len': (lambda x,a,b: len(x)<a or len(x)>b),
        'both_len': (lambda x,a,b: x.src_len<a or x.src_len>b or x.trg_len<a or x.trg_len>b),
        'src_len': (lambda x,a,b: x.src_len<a or x.src_len>b), 'trg_len': (lambda x,a,b: x.trg_len<a or x.trg_len>b),
    }
    # --

    def __init__(self, conf: ZDataBatcherConf):
        self.conf = conf
        # --
        self.batch_size_f = ZDataBatcher._LEN_F.get(conf.batch_size_f)
        if self.batch_size_f is None:
            self.batch_size_f = eval(conf.batch_size_f)
        self.sort_size_f = ZDataBatcher._LEN_F.get(conf.sort_size_f)
        if self.sort_size_f is None:
            self.sort_size_f = eval(conf.sort_size_f)
        _dump_f = ZDataBatcher._DUMP_F.get(conf.filter_dump_f)
        if _dump_f is None:
            self.dump_f = eval(conf.filter_dump_f)
        else:
            self.dump_f = (lambda x: _dump_f(x, conf.filter_min_length, conf.filter_max_length))
        # --

    def get_batched_streamer(self, input_stream, dataset):
        conf = self.conf
        batch_arranger = BatchArranger(
            input_stream, batch_size=conf.batch_size, maxi_bsize=conf.batch_maxi_bsize, batch_size_f=self.batch_size_f,
            dump_detectors=[self.dump_f], sorting_keyer=self.sort_size_f, shuffle_batches_times=conf.bucket_shuffle_times)
        ret = WrapperStreamer(batch_arranger, func=(lambda insts: DataBatch(insts, dataset)))
        return ret
        # --
# --

# --
class ZDatasetConf(Conf):
    def __init__(self):
        # info
        self.data_name = ""
        self.data_sample_rate = SVConf.direct_conf(val=1.)  # sample by rate
        self.data_tasks = []  # tasks to perform!
        self.data_eval_weight = 1.
        # reader / writer
        self.R = MultiFileReaderConf()
        self.W = JsonWriterConf()
        # prep
        self.preprocessors = []  # need to slightly modify the data?
        # cache?
        self.cache_conf: CacheStreamerConf = ConfEntryChoices({'yes': CacheStreamerConf(), 'no': None}, 'no')
        # batch
        self.batch_conf = ZDataBatcherConf()

    def has_files(self):
        return len(self.R.input_paths) > 0

class ZDataset:
    def __init__(self, conf: ZDatasetConf, wset: str):
        self.conf = conf
        self.name = conf.data_name
        self.wset = wset
        self.tasks = OrderedDict()  # tasks to perform!
        for t in conf.data_tasks:
            self.tasks[t] = 1
        # precessors
        self._preprocessors = [ZDataPreprocessor.try_load_and_lookup(z).T for z in conf.preprocessors]
        # batcher
        self._batcher = ZDataBatcher(conf.batch_conf)
        # --
        # build the pipe
        zlog(f"Build the pipe of {self} from {self.conf.R.input_paths} with {self.tasks}")
        self.pipe_reader = MultiFileReader(self.conf.R)
        self.pipe_preper = WrapperStreamer(self.pipe_reader, func=self._do_prep, ignore_none=True)
        if conf.cache_conf is not None:
            self.pipe_cacher = CacheStreamer(self.conf.cache_conf, self.pipe_preper)
        else:
            self.pipe_cacher = self.pipe_preper
        self.pipe_batcher = self._batcher.get_batched_streamer(self.pipe_cacher, self)
        # --

    def __repr__(self):
        return f"Dataset({self.name}:{self.wset})"

    def _do_prep(self, one):
        for pp in self._preprocessors:
            one = pp(one, self)
            if one is None:
                break
        return one

    # add extra preprocessors, usually indexers from the model
    def add_preprocessors(self, ps: List):
        self._preprocessors.extend(ps)

    # only yielding raw insts
    def yield_insts(self):
        reader = MultiFileReader(self.conf.R)
        yield from reader

    # gold insts for eval; todo(note): currently simply assume the same as input!
    def yield_gold_insts(self):
        yield from self.yield_insts()

    # yielding batches from the pipe: reader -> preper -> (cache) -> batcher
    def yield_batches(self, loop=None):
        if loop is None:  # auto decide it by wset name
            loop = (self.wset == "train")
        # --
        _loop_trg = -1 if loop else 1
        _ii = 0
        while _ii != _loop_trg:
            _ii += 1
            with Timer(info=f"{self} Epoch[{_ii}/{_loop_trg}]"):
                yield from self.pipe_batcher
        # --

    # write outputs
    def write_insts(self, insts, extra_suffix=''):
        if self.conf.W.output_path:
            with JsonWriter(self.conf.W, fd_or_path=self.conf.W.output_path + extra_suffix) as writer:
                writer.dump_iter(insts)

# --
# b zgen/data/dataset:110
