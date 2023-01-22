#

# some helpers

__all__ = [
    "CmdLineConf", "CmdLineParser", "RecordNode", "RecordNodeVisitor",
]

import configparser
import sys
from typing import Dict, List, Callable, Union
from shlex import split as sh_split
import traceback
from typing import List
from zgen.utils import Conf, zlog, zfatal, zopen
from zgen.data.stream import Streamer
from zgen.utils import IdAssignable

# =====
# command-line parser (reader)

class CmdLineConf(Conf):
    def __init__(self):
        self.cmd_input = ""  # by default from input
        self.assign_target = True  # allow "[target-name=]args..."
        self.target_sep = "="  # sep target and cmd
        self.kwargs_sep = "--"  # sep args and kwargs
        self.kwargs_kv_sep = ":"  # sep k:v for kwargs

class CmdLineParser(Streamer):
    def __init__(self, conf: CmdLineConf, **kwargs):
        super().__init__()
        self.conf = conf.direct_update(**kwargs)
        # --
        if self.conf.cmd_input in ["", "-"]:
            self.streamer = None
        else:
            raise NotImplementedError()
        # --

    def _iter_line(self):
        conf = self.conf
        if conf.cmd_input in ["", "-"]:
            while True:
                # read from input
                try:
                    line = input(">> ")
                    if line.strip() == "":
                        continue
                    yield line
                except EOFError:
                    break
                except KeyboardInterrupt:
                    continue
        else:
            with zopen(conf.cmd_input) as fd:
                for line in fd:
                    if line.strip() != "":
                        yield line

    def _iter(self):
        conf = self.conf
        for line in self._iter_line():
            target, args, kwargs = None, [], {}
            cmd = line.strip()
            # find target
            if conf.assign_target:
                tmp_fields = cmd.split(conf.target_sep, 1)
                if len(tmp_fields) == 2:
                    tmp_target, remainings = [x.strip() for x in tmp_fields]
                    if str.isidentifier(tmp_target):  # make sure it is identifier
                        target = tmp_target  # assign target
                        line = remainings
            # split into args
            try:  # try shell splitting
                tmp_args = sh_split(line)
                cur_i = 0
                # collect *args
                while cur_i < len(tmp_args):
                    cur_a = tmp_args[cur_i]
                    if cur_a == conf.kwargs_sep:
                        cur_i += 1  # skip this one!
                        break
                    else:
                        args.append(cur_a)
                    cur_i += 1
                # collect **kwargs
                while cur_i < len(tmp_args):
                    _k, _v = tmp_args[cur_i].split(conf.kwargs_kv_sep)
                    kwargs[_k] = _v
                    cur_i += 1
                yield (cmd, target, args, kwargs)
            except:
                zlog(f"Err in CMD-Parsing: {traceback.format_exc()}")

# =====

# todo(note): node_id and ch_key are bonded together!!
class TreeNode(IdAssignable):
    def __init__(self, id=None, **kwargs):
        if id is None:  # automatic id
            self.id = self.__class__.get_new_id()
        else:
            self.id = id
        # --
        self.par: TreeNode = None
        self.chs_map: Dict = {}  # id -> node
        # extra properties
        self.props = kwargs

    # has ch?
    def has_ch(self, node: 'TreeNode'):
        return node.id in self.chs_map

    # get ch?
    def get_ch(self, id, df=None):
        return self.chs_map.get(id, df)

    # add one children
    def add_ch(self, node: 'TreeNode'):
        assert node.par is None, "Node already has parent!"
        assert node.id not in self.chs_map, "Node already in chs_map"
        self.chs_map[node.id] = node
        node.par = self  # link both ways

    # detach one child from self
    def detach_ch(self, node: 'TreeNode'):
        assert node.par is self
        del self.chs_map[node.id]
        node.par = None

    # detach from parent
    def detach_par(self):
        self.par.detach_ch(self)

    # =====
    # like a dictionary
    def __getitem__(self, item):
        return self.chs_map[item]

    def __contains__(self, item):
        return item in self.chs_map

    def keys(self):
        return self.chs_map.keys()

    def values(self):
        return self.chs_map.values()

    def is_root(self):
        return self.par is None

    def __getattr__(self, item):
        if item in self.props:
            return self.props[item]
        else:
            raise AttributeError()

    # get nodes of descendents
    def get_descendants(self, recursive=True, key: Union[str,Callable]=None, preorder=True, include_self=True):
        _chs_f = TreeNode.get_enum_f(key)
        # --
        def _get_descendants(_n: 'TreeNode'):
            _ret = [_n] if (include_self and preorder) else []
            ch_list = _chs_f(_n)
            if recursive:
                for _n2 in ch_list:
                    _ret.extend(_get_descendants(_n2))
            else:  # only adding direct ones
                _ret.extend(ch_list)
            if include_self and (not preorder):
                _ret.extend(_n)
            return _ret
        # --
        return _get_descendants(self)

    # get parent, grandparent, etc; h2l means high to low
    def get_antecedents(self, max_num=-1, include_self=False, h2l=True):
        ret, cur_num = [], 0
        _cur = self if include_self else self.par
        while cur_num != max_num and _cur is not None:
            ret.append(_cur)
            _cur = _cur.par
            cur_num += 1
        if h2l:  # high-to-low?
            ret.reverse()
        return ret

    # helper for enum children
    @staticmethod
    def get_enum_f(key: Union[str, Callable]=None):
        # prepare get_values
        if not isinstance(key, Callable):
            if key is None:
                _chs_f = lambda x: x.chs_map.values()  # order does not matter!
            else:
                assert isinstance(key, str)
                _chs_f = lambda x: sorted(x.chs_map.values(), key=lambda v: getattr(v, key))
        else:
            _chs_f = key
        return _chs_f

    # recursively visit (updating)
    def rec_visit(self, visitor: 'TreeNodeVisitor'):
        # pre-visit self
        pre_value = visitor.pre_visit(self)
        # first visit all the children nodes
        ch_values = [n.rec_visit(visitor) for n in visitor.enum_chs(self)]
        # post-visit self
        return visitor.post_visit(self, pre_value, ch_values)

# -----
class TreeNodeVisitor:
    # which order to see chs: by default no specific order
    def enum_chs(self, node: TreeNode):
        return node.values()

    # pre_visit: by default do nothing
    def pre_visit(self, node: TreeNode):
        return None

    # post_visit (with chs's values)
    def post_visit(self, node: TreeNode, pre_value, ch_values: List):
        raise NotImplementedError()

# =====
# record node

class RecordNode(TreeNode):
    def __init__(self, par: 'RecordNode', path: List, **kwargs):
        super().__init__(id=('R' if len(path)==0 else path[-1]), **kwargs)
        # --
        # path info
        self.path = tuple(path)
        self.name = ".".join([str(z) for z in path])
        self.level = len(path)  # starting from 0
        # content info
        self.count = 0  # all the ones that go through this node
        self.count_end = 0  # only those ending at this node
        self.objs: List = []  # only added to the end points!
        # add par
        if par is not None:
            par.add_ch(self)

    @classmethod
    def new_root(cls):
        return cls(None, [])

    # =====
    # recording a seq and add/extend node if needed
    def record_seq(self, seq, count=1, obj=None):
        assert self.is_root(), "Currently only support adding from ROOT"
        # make it iterable
        if not isinstance(seq, (list, tuple)):
            seq = [seq]
        # recursive adding
        cur_node = self
        cur_path = []
        while True:
            # update for current node
            cur_node.count += count
            if obj is not None:
                cur_node.objs.append(obj)
            # next one
            if len(seq) <= 0:
                cur_node.count_end += count
                break
            seq0, seq = seq[0], seq[1:]
            cur_path.append(seq0)
            next_node = cur_node.get_ch(seq0)  # try to get children
            if next_node is None:
                next_node = RecordNode(cur_node, cur_path)  # no need copy, since changed to a new tuple later.
            cur_node = next_node

    # content for printing
    def get_content(self):
        return None

class RecordNodeVisitor(TreeNodeVisitor):
    pass
