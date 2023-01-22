#

# call eval

from zgen.utils import Conf, init_everything, zlog, OtherHelper
from zgen.data import MultiFileReaderConf, MultiFileReader
from .eval import *

# --
class MainConf(Conf):
    def __init__(self):
        self.gold = MultiFileReaderConf()
        self.pred = MultiFileReaderConf()
        self.eval = SeqEvalerConf.direct_conf(eval_seq_name_gold='plain', eval_seq_name_pred='plain')

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    gold_insts = list(MultiFileReader(conf.gold))
    pred_insts = list(MultiFileReader(conf.pred))
    evaler = SeqEvaler(conf.eval)
    res = evaler.eval(gold_insts, pred_insts)
    zlog(f"Eval [g={conf.gold.input_paths},p={conf.pred.input_paths}]: res = ")
    OtherHelper.printd(res)
    # --

# PYTHONPATH=?? python3 -m zgen.eval.main gold.input_paths:?? pred.input_paths:??
# ... evals:sacrebleu sacrebleu_trg_lang:zh 'sacrebleu_cmd_opt:-t wmt18 -l en-zh'
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
