#

__all__ = [
    "Streamer", "WrapperStreamer", "IterStreamer", "BatchArranger",
    "CacheStreamerConf", "CacheStreamer",
    "Dumper",
]

# --
import os
import pickle
from typing import Union, Iterable, Callable, List
from zgen.utils import Random, Constants, Conf, zopen, zwarn, zlog

STREAMER_RANDOM_GEN = Random.get_generator('stream')

# basic streamer
class Streamer:
    def __init__(self):
        self._iterator = None

    def __iter__(self):
        self._iterator = self._iter()
        return self

    def __next__(self):
        return next(self._iterator)

    def _iter(self):
        raise NotImplementedError()

# Streamer wrapper/decorator, stacked streamers, driven by the ended Streamer
class WrapperStreamer(Streamer):
    def __init__(self, base_streamer: Streamer, func: Callable = None, inplaced=False, ignore_none=False):
        super().__init__()
        self.base_streamer: Streamer = base_streamer
        self.func = func
        self.inplaced = inplaced
        self.ignore_none = ignore_none

    def _iter(self):  # by default simply yield from base
        ff = self.func
        if ff is None:
            yield from self.base_streamer._iter()
        else:
            _inplaced = self.inplaced
            _accept_none = not self.ignore_none
            for one in self.base_streamer._iter():
                z = ff(one)
                if _inplaced:
                    yield one
                elif _accept_none or (z is not None):
                    yield z

# from iterable (or lambda: iterable)
class IterStreamer(Streamer):
    def __init__(self, src: Union[Iterable, Callable], is_f=False):
        super().__init__()
        self.src = src
        self.is_f = is_f

    def _iter(self):
        src = self.src() if self.is_f else self.src
        yield from src

# handling the batching of instances, also filtering, sorting, recording, etc.
# streamer: base, batch_size: sum(batch_size_f(z) for z), maxibatch_size: read-in bs*mbs every time,
# dump_detectors, single_detectors, sorting_keyer: sort bs*mbs, shuffling: shuffle on buckets in bs*mbs
class BatchArranger(WrapperStreamer):
    def __init__(self, base_streamer: Streamer, batch_size: int, maxi_bsize: int, batch_size_f: Callable = None,
                 dump_detectors: Union[Callable, List[Callable]] = None, single_detectors: Union[Callable, List[Callable]] = None,
                 sorting_keyer: Callable = None, shuffle_batches_times=0):
        super().__init__(base_streamer)
        self.batch_size = batch_size
        self.batch_size_f = (lambda one: 1) if batch_size_f is None else batch_size_f
        # todo(note): if <=0 then read all at one time and possibly sort all
        self.maxibatch_size = maxi_bsize if maxi_bsize>0 else Constants.INT_PRAC_MAX
        if dump_detectors is not None and not isinstance(dump_detectors, Iterable):
            dump_detectors = [dump_detectors]
        self.dump_detectors = [] if dump_detectors is None else dump_detectors
        if single_detectors is not None and not isinstance(single_detectors, Iterable):
            single_detectors = [single_detectors]
        self.single_detectors = [] if single_detectors is None else single_detectors
        self.sorting_keyer = sorting_keyer  # default(None) no sorting
        self.shuffle_batches_times = shuffle_batches_times  # shuffle batches inside the maxi-batches?

    def _iter(self):
        buffered_bsize = 0
        buffer = []  # list of instances
        buckets = []  # list of already prepared batch of instances
        _K = self.batch_size * self.maxibatch_size
        # --
        def _yield_them():
            if buffered_bsize > 0:
                new_buckets = BatchArranger.group_buckets(
                    buffer, thresh_all=self.batch_size, size_f=self.batch_size_f, sort_key=self.sorting_keyer)
                buckets.extend(new_buckets)
                # shuffle?
                for _ in range(self.shuffle_batches_times):
                    STREAMER_RANDOM_GEN.shuffle(buckets)
                # yield
                yield from buckets
            # clear
            buffer.clear()
            buckets.clear()
        # --
        # fetching loop
        for one in self.base_streamer._iter():
            # dump instances (like short or long instances)
            dump_instance = any(f_(one) for f_ in self.dump_detectors)
            if dump_instance:
                continue
            # single instances
            single_instance = any(f_(one) for f_ in self.single_detectors)
            if single_instance:
                # make this a singleton
                buckets.append([one])
            else:
                # add to buffer
                buffer.append(one)
            buffered_bsize += self.batch_size_f(one)
            # ready to push?
            if buffered_bsize >= _K:
                yield from _yield_them()
                buffered_bsize = 0
        # final batch
        yield from _yield_them()
        buffered_bsize = 0
        # --

    @staticmethod
    def group_buckets(input_insts: List, thresh_all: int = None, thresh_diff: int = None,
                      size_f: Callable = (lambda x: len(x)), sort_key: Callable = None):
        # --
        # special case for single inst
        if thresh_all <= 1:
            return [[z] for z in input_insts]
        # --
        if thresh_all is None:
            thresh_all = Constants.INT_PRAC_MAX
        if thresh_diff is None:
            thresh_diff = Constants.INT_PRAC_MAX
        # sort inputs?
        if sort_key is not None:
            input_insts = sorted(input_insts, key=sort_key)
        # prepare buckets
        buckets = []
        cur_size_all, cur_size_start = 0, None
        tmp_bucket = []
        for one in input_insts:
            one_size = size_f(one)
            if len(tmp_bucket) == 0:  # always add when empty
                cur_size_all = cur_size_start = one_size
                tmp_bucket = [one]  # todo(+w): here also need to check thresh_all!!
            elif one_size-cur_size_start >= thresh_diff:  # a new start with current
                buckets.append(tmp_bucket)
                cur_size_all = cur_size_start = one_size
                tmp_bucket = [one]
            else:
                cur_size_all += one_size  # add to cur_size
                tmp_bucket.append(one)  # add one
                if cur_size_all >= thresh_all:  # a new start after current
                    buckets.append(tmp_bucket)
                    cur_size_all, cur_size_start = 0, None
                    tmp_bucket = []
        # --
        if len(tmp_bucket) > 0:
            buckets.append(tmp_bucket)
        return buckets

# --
class CacheStreamerConf(Conf):
    def __init__(self):
        self.cache_file = "./_zcache"
        self.mem_cap = 50000  # capacity (num of insts) in memory
        self.shuffle_times = 1  # shuffle inside each bucket

    def _do_validate(self):
        from zgen.utils import nn as nnutils
        if nnutils.use_ddp() and self.cache_file.lower() != "__mem__":
            _rank, _wsize = nnutils.ddp_rank(), nnutils.ddp_world_size()
            self.cache_file = self.cache_file + str(_rank)  # simply make a different name!
            # zlog(f"Change cache.cache_file to {self.cache_file}")

class CacheStreamer(WrapperStreamer):
    def __init__(self, conf: CacheStreamerConf, base_streamer: Streamer):
        super().__init__(base_streamer)
        # --
        self.conf = conf
        # --
        # use in-mem cache?
        self.c = []
        # --

    def _iter(self):
        conf = self.conf
        _cap = conf.mem_cap
        # --
        def _yield_them(_caches):
            for _ii in range(conf.shuffle_times):
                STREAMER_RANDOM_GEN.shuffle(_caches)
            yield from _caches
            _caches.clear()
        # --
        # todo(+N): not a good design ...
        if conf.cache_file.lower() == '__mem__':
            if self.c:
                caches = self.c.copy()
                if len(caches) > _cap:
                    zwarn(f"Mem cache larger than cap: {len(caches)}>{_cap}")
                yield from _yield_them(caches)
            else:
                caches = []
                for one in self.base_streamer._iter():
                    caches.append(one)
                    yield one
                if len(caches) > _cap:
                    zwarn(f"Mem cache larger than cap: {len(caches)}>{_cap}")
                self.c = caches
        else:
            from .inst import DataInst
            # read from cache file!
            if os.path.exists(conf.cache_file):
                # read from cache
                with zopen(conf.cache_file, 'rb') as fd:
                    caches = []
                    _eof = False
                    while not _eof:
                        while len(caches) < _cap:
                            try:
                                v = pickle.load(fd)
                                inst = DataInst.create_from_json(v)
                                caches.append(inst)
                            except EOFError:
                                _eof = True
                                break
                        yield from _yield_them(caches)
            else:  # first time read from base_streamer
                with zopen(conf.cache_file, 'wb') as fd:
                    caches = []
                    for one in self.base_streamer._iter():
                        caches.append(one)
                        if len(caches) >= _cap:
                            for inst in caches:
                                v = inst.to_json(minimal=True)
                                pickle.dump(v, fd)
                            yield from _yield_them(caches)
                    for inst in caches:
                        v = inst.to_json(minimal=True)
                        pickle.dump(v, fd)
                    yield from _yield_them(caches)
            # --

# just make it simple
class Dumper:
    def dump_one(self, obj: object):
        raise NotImplementedError()

    def dump_iter(self, iter: Iterable):
        for one in iter:
            self.dump_one(one)

    def close(self):
        pass

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()

# --
# b zgen/data/stream:111
