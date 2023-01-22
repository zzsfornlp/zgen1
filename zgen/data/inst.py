#

# data instance (data point)

__all__ = [
    "data_reg", "DataInst", "SeqDataInst", "DataBatch", "inst_obtain_getter_setter",
]

from typing import List, Type

# --

class DataInst:
    _Type2Name = {}
    _Name2Type = {}

    def __init__(self):
        self._idx = None
        self.info = {}

    @staticmethod
    def reg_type_name(t: Type, n: str):
        assert t not in DataInst._Type2Name and n not in DataInst._Name2Type
        DataInst._Type2Name[t] = n
        DataInst._Name2Type[n] = t

    @staticmethod
    def type2name(t: Type, df=None):
        return DataInst._Type2Name.get(t, df)

    @staticmethod
    def name2type(n: str, df=None):
        return DataInst._Name2Type.get(n, df)

    def to_json(self, minimal=False):
        ret = {'_t': DataInst.type2name(self.__class__)}
        if not minimal and self._idx is not None:
            ret['_idx'] = self._idx
        if self.info:
            ret['info'] = self.info
        return ret

    def from_json(self, v):
        assert DataInst.name2type(v['_t']) is self.__class__
        if 'info' in v:
            self.info = v['info']
        if '_idx' in v:
            self._idx = v['_idx']

    @staticmethod
    def create_from_json(v):
        t = DataInst.name2type(v['_t'])
        ret = t()
        ret.from_json(v)
        return ret

# reg for both conf and node
def data_reg(name: str):
    def _f(cls: Type):
        DataInst.reg_type_name(cls, name)
        return cls
    return _f

@data_reg('seq')
class SeqDataInst(DataInst):
    def __init__(self, tokens: List[str] = None):
        super().__init__()
        # src tokens
        self.tokens = tokens
        self.idxes: List[int] = None
        # optional trg tokens
        self.trg_tokens = None
        self.trg_idxes: List[int] = None
        # --

    @property
    def src_len(self):
        return len(self.tokens) if self.idxes is None else len(self.idxes)

    @property
    def trg_len(self):
        _items = self.trg_tokens if self.trg_idxes is None else self.trg_idxes
        return None if _items is None else len(_items)

    def __len__(self):  # by default trg length!
        ret = self.trg_len
        return self.src_len if ret is None else ret

    def __repr__(self):
        return f"Seq:{self.tokens}"

    def to_json(self, minimal=False):
        v = super().to_json()
        v['idxes'] = self.idxes
        v['trg_idxes'] = self.trg_idxes
        if not minimal:
            v['tokens'] = self.tokens
            v['trg_tokens'] = self.trg_tokens
        return v

    def from_json(self, v):
        super().from_json(v)
        self.idxes = v['idxes']
        self.trg_idxes = v['trg_idxes']
        if 'tokens' in v:
            self.tokens = v['tokens']
        if 'trg_tokens' in v:
            self.trg_tokens = v['trg_tokens']

# --
# helper for getter and setter
def _inst_getter_plain(inst):
    return inst.tokens, inst.idxes
def _inst_setter_plain(inst, tokens, idxes):
    inst.tokens, inst.idxes = tokens, idxes
def _inst_getter_trg(inst):
    return inst.trg_tokens, inst.trg_idxes
def _inst_setter_trg(inst, tokens, idxes):
    inst.trg_tokens, inst.trg_idxes = tokens, idxes
def inst_obtain_getter_setter(seq_name: str):
    return {"plain": (_inst_getter_plain, _inst_setter_plain),
            'trg': (_inst_getter_trg, _inst_setter_trg)}[seq_name]
# --

class DataBatch:
    def __init__(self, insts: List[DataInst], dataset):
        self.insts = insts
        self.dataset = dataset

    def __len__(self):
        return len(self.insts)

    def __repr__(self):
        return f"DataBatch(N={len(self)})"
