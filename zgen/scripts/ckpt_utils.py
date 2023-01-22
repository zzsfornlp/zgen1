#

# some tools to manipulate ckpt models!

import sys
import json
from collections import Counter, OrderedDict
import re
import torch
from zgen.utils import Conf, init_everything, zopen, zlog, Random

class MainConf(Conf):
    def __init__(self):
        # input
        self.in0 = ""
        self.in1 = ""
        # --
        self.action = ''
        # a1: extract model
        self.extract_filter = ''  # re pattern
        self.extract_sub = ''  # '...=>...'
        self.extract_out = ""
        # a2: compare m0 & m1
        self.cmp_break = False

# --
def _load_model(f: str):
    if f:
        m = torch.load(f, map_location='cpu')
        zlog(f"Load from {f}")
    else:
        m = None
    return m
def _save_model(m, f: str):
    if f:
        torch.save(m, f)
        zlog(f"Save to {f}")
# --
def _extract_model(conf):
    m = _load_model(conf.in0)
    # step 1: filter
    pat = re.compile(conf.extract_filter)
    orig_keys = list(m.keys())
    del_keys = []
    for k in orig_keys:
        if pat.fullmatch(k):
            pass
        else:
            del m[k]
            del_keys.append(k)
    zlog(f"Filter by ``{conf.extract_filter}'': del={len(del_keys)},kept={len(m)}")
    # step 2: change name
    if conf.extract_sub:
        s1, s2 = conf.extract_sub.split("=>")
        zlog(f"Sub with {s1} => {s2}")
        m_new = OrderedDict()
        for k, v in m.items():
            k_new = re.sub(s1, s2, k)
            m_new[k_new] = v
            if k != k_new:
                zlog(f"Change name {k} => {k_new}")
        m = m_new
    # save
    _save_model(m, conf.extract_out)
    # --
# --
def _cmp_model(conf):
    m0, m1 = _load_model(conf.in0), _load_model(conf.in1)
    keys0, keys1 = set(m0.keys()), set(m1.keys())
    # extra ones?
    extra0, extra1 = keys0-keys1, keys1-keys0
    zlog(f"Extra in M0 [{len(extra0)}/{len(keys0)}]: {extra0}")
    zlog(f"Extra in M1 [{len(extra1)}/{len(keys1)}]: {extra1}")
    # common ones
    diffs = []
    for k in sorted(keys0 & keys1):
        v0, v1 = m0[k], m1[k]
        if not torch.allclose(v0, v1):
            if conf.cmp_break:
                breakpoint()
            diffs.append(k)
    zlog(f"Diff are [{len(diffs)}]: {diffs}")
# --

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # go!!
    action_map = {'extract': _extract_model, 'cmp': _cmp_model}
    zlog(f"Go with action: {conf.action}")
    action_map[conf.action](conf)
    # --

# PYTHONPATH=../src/ python3 ckpt_utils.py
if __name__ == '__main__':
    main(*sys.argv[1:])

"""
# cmp
PYTHONPATH=../src/ python3 -m zgen.scripts.ckpt_utils action:cmp in0:./zmodel.curr.m0 in1:./zmodel.curr.m1 cmp_break:1
# extract
PYTHONPATH=../src/ python3 -m pdb ckpt_utils.py action:extract in0:./zmodel.curr.m0 'extract_filter:^Menc.*' 'extract_sub:Menc.=>' 'extract_out:_tmp.m'
"""
