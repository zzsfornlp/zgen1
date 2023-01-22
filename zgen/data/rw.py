#

# reader & writer

__all__ = [
    "DataReaderConf", "DataReader", "MultiFileReaderConf", "MultiFileReader",
    "DataWriterConf", "DataWriter", "JsonWriterConf", "JsonWriter",
]

from typing import List
import json
import pickle
import numpy as np
from zgen.utils import Registrable, Conf, Timer, zopen, zopen_withwrapper, zlog, zglob, Random, zwarn
from zgen.utils import nn as nnutils
from .inst import DataInst, SeqDataInst
from .stream import Streamer, Dumper

# --
# readers

# base class
class DataReaderConf(Conf):
    pass

class DataReader(Streamer):
    def __init__(self, conf: DataReaderConf):
        super().__init__()
        self.conf = conf
        # --

# --
# plain txt
def _read_plain(files: List):
    ii = 0
    for file in files:
        with zopen_withwrapper(file) as fd:
            for line in fd:
                tokens = line.split()
                if len(tokens) == 0:
                    zwarn(f"Read an empty line from {file}")
                ret = SeqDataInst(tokens)
                ret._idx = ii  # note: set order!
                ii += 1
                yield ret
# --
# json
def _read_json(files: List):
    ii = 0
    for file in files:
        with zopen_withwrapper(file) as fd:
            for line in fd:
                v = json.loads(line)
                ret = DataInst.create_from_json(v)
                ret._idx = ii  # note: set order!
                ii += 1
                yield ret
# --
# pkl
def _read_pkl(files: List):
    ii = 0
    for file in files:
        with zopen_withwrapper(file, mode='rb') as fd:
            while True:
                try:
                    ret = pickle.load(fd)
                    yield ret  # directly return this!
                except EOFError:
                    break
# --
# always yield None
def _yield_none():
    while True:
        yield None
# --

# multiple files reader
class MultiFileReaderConf(DataReaderConf):
    def __init__(self):
        super().__init__()
        self.input_dir = ""
        self.input_paths = []
        self.input_format = 'plain'
        self.split_ddp = True
        self.shuffle_path_times = 0
        # take certain
        self.take_first = -1  # if>0
        # bitext options
        self.is_bitext = False  # bitext?
        self.bitext_src_suffix = ""
        self.bitext_trg_suffix = ""
        # target aux info option
        self.aux_suffix = ""
        # --

    def _do_validate(self):
        # first do zglob: note: only check src!
        _suffix = self.bitext_src_suffix if self.is_bitext else ''
        real_paths = sum([zglob(self.input_dir+p+_suffix, assert_exist=False, check_iter=10, sort=True)
                          for p in self.input_paths], [])
        if len(self.input_paths) > 0 and len(real_paths) <= 0:
            zwarn(f"Failed to find anything for {self.input_dir} + {self.input_paths} + {_suffix}")
        from zgen.utils import nn as nnutils
        if nnutils.use_ddp() and self.split_ddp:
            _rank, _wsize = nnutils.ddp_rank(), nnutils.ddp_world_size()
            real_paths = [p for i,p in enumerate(real_paths) if (i%_wsize==_rank)]
        self.input_paths = real_paths

class MultiFileReader(DataReader):
    def __init__(self, conf: MultiFileReaderConf):
        super().__init__(conf)
        self.readf = {'plain': _read_plain, 'json': _read_json}[conf.input_format]
        # --

    def _read_files(self, files):
        conf: MultiFileReaderConf = self.conf
        if conf.is_bitext:  # yield bitext
            for src_file in files:
                file_prefix = src_file[:-len(conf.bitext_src_suffix)] if conf.bitext_src_suffix else src_file
                trg_file = file_prefix + conf.bitext_trg_suffix
                # --
                gen_src, gen_trg = self.readf([src_file]), self.readf([trg_file])
                gen_aux = _read_pkl([file_prefix+conf.aux_suffix]) if conf.aux_suffix else _yield_none()
                for inst, inst2, aux in zip(gen_src, gen_trg, gen_aux):
                    inst.trg_tokens = inst2.tokens  # simply re-assign!
                    inst._aux = aux  # directly assign as '_aux'
                    yield inst
        else:
            yield from self.readf(files)
        # --

    def _iter(self):
        # --
        conf: MultiFileReaderConf = self.conf
        files = list(conf.input_paths)
        if len(files) > 0 and conf.shuffle_path_times>0:
            _gen = Random.get_generator(f'r{nnutils.ddp_rank()}')
            for _ in range(conf.shuffle_path_times):
                _gen.shuffle(files)
        # --
        zlog(f"DataIter(cc={conf.take_first}x{nnutils.ddp_world_size()}={conf.take_first*nnutils.ddp_world_size()}) with {files}")
        cc = 0
        for inst in self._read_files(files):
            yield inst
            cc += 1
            if cc == conf.take_first:
                break
        # --

# --
# writers

# base class
class DataWriterConf(Conf):
    pass

class DataWriter(Dumper):
    def __init__(self, conf: DataWriterConf):
        self.conf = conf
        # --

# simple one
class JsonWriterConf(DataWriterConf):
    def __init__(self):
        super().__init__()
        self.output_path = ""

class JsonWriter(DataWriter):
    def __init__(self, conf: DataWriterConf, fd_or_path=None):
        super().__init__(conf)
        self.fd_or_path = fd_or_path
        if fd_or_path is None:
            fd_or_path = conf.output_path
        if isinstance(fd_or_path, str):
            self.should_close = True
            self.fd = zopen(fd_or_path, 'w')
        else:
            self.should_close = False
            self.fd = fd_or_path
        # --

    def close(self):
        if self.should_close:  # only close if that is what we opened
            self.fd.close()

    def dump_one(self, inst):
        ss = json.dumps(inst.to_json()) + "\n"
        self.fd.write(ss)

# --
# b zgen/data/rw:67
