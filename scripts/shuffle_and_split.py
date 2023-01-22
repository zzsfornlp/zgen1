#

# shuffle (and split) training data

import os
from collections import Counter
from zgen.utils import Conf, init_everything, zopen, zlog, Random

class MainConf(Conf):
    def __init__(self):
        self.input_paths = []
        self.output_paths = []
        self.split_pieces = 16
        # --
        self.min_len = 0
        self.max_len = 100000
        self.shuffle_times = 1
        # --

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    assert len(conf.input_paths) == len(conf.output_paths)
    cc = Counter()
    # --
    # first read all parallel lines
    all_insts = []
    for ii, file_in in enumerate(conf.input_paths):
        with zopen(file_in) as fd:
            if ii == 0:
                all_insts = [[line] for line in fd]
            else:
                count = 0
                for jj, line in enumerate(fd):
                    all_insts[jj].append(line)
                    count += 1
                assert count == len(all_insts)
    zlog(f"Read all {len(all_insts)} x {len(conf.input_paths)}")
    # shuffle
    _gen = Random.get_generator('shuffle')
    for _ in range(conf.shuffle_times):
        _gen.shuffle(all_insts)
    # split and write
    s_pat = f"%0{len(str(conf.split_pieces))}d" if conf.split_pieces>0 else None
    for ss in range(conf.split_pieces):
        for ii, file_out in enumerate(conf.output_paths):
            # fix path
            p0, p1 = os.path.dirname(file_out), os.path.basename(file_out)
            _fields = p1.rsplit('.', 1)
            if len(_fields) < 2: _fields = [''] + _fields
            # p1 = f'{ss}.'.join(_fields)
            p1 = _fields[0] + ((s_pat%ss) if s_pat is not None else "") + "." + _fields[1]
            pp = os.path.join(p0, p1)
            with zopen(pp, 'w') as fd:
                count = 0
                for jj in range(ss, len(all_insts), conf.split_pieces):
                    fd.write(all_insts[jj][ii])
                    count += 1
            zlog(f"Write to {pp}(ss={ss},ii={ii}): Num = {count}")
    # --

# --
# extra helper
def check_lengths(f: str):
    from collections import Counter
    cc = Counter()
    with open(f) as fd:
        for line in fd:
            _len = len(line.split())
            cc[_len] += 1
    print(cc)
    print({k:v for k,v in cc.items() if k>=100})
# --

# PYTHONPATH=../src/ python3 shuffle_and_split.py input_paths:IN output_paths:OUT
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
