#

# wrapper for tokenizer (vocab)

__all__ = [
    "ZToker", "ZTokerVWrapper",
]

from zgen.data import SimpleVocab
from zgen.utils import Conf

class ZToker:
    def __init__(self, base):
        self.base = base

    # simply wrapping
    def __getattr__(self, name):
        return getattr(self.base, name)

class ZTokerVWrapper(ZToker):
    def __init__(self, vocab: SimpleVocab):
        super().__init__(vocab)
        self._full_voc = None  # cached
        # --

    def convert_tokens_to_ids(self, tokens):
        return self.base.seq_word2idx(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.base.seq_idx2word(ids)

    def get_vocab(self):
        if self._full_voc is None:
            voc = {}
            for ii, vv in enumerate(self.base.full_i2w):
                voc[vv] = ii
            self._full_voc = voc
        return self._full_voc

    def __call__(self, *args, **kwargs):
        raise RuntimeError()  # no such function!

    # special ones
    @property
    def cls_token_id(self): return self.base.bos

    @property
    def sep_token_id(self): return self.base.eos

    @property
    def pad_token_id(self): return self.base.pad

    @property
    def mask_token_id(self): return self.base.mask

    @property
    def unk_token_id(self): return self.base.unk

    @property
    def vocab_size(self): return len(self.base)

    def __repr__(self):
        return f"ZTokerVWrapper of {self.base}"
