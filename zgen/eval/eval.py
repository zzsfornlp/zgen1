#

# evaler

__all__ = [
    "SeqEvalerConf", "SeqEvaler",
]

import os
import re
from typing import List
from zgen.utils import nn as nnutils
from zgen.utils import Conf, zopen, system, zlog
from zgen.data import inst_obtain_getter_setter, VocabHelper

class SeqEvalerConf(Conf):
    def __init__(self):
        self.evals = ['bleu', 'cm']  # ['bleu', 'sacrebleu', 'rouge', 'cm']
        # common options
        self.eval_seq_name_gold = 'trg'  # eval which one?
        self.eval_seq_name_pred = 'trg'  # eval which one?
        self.delete_toks = ['[CLS]', '[SEP]', '[MASK]', '[PAD]', '<s>', '</s>'] \
                           + [VocabHelper.convert_special_pattern(z) for z in ['bos', 'eos', 'pad', 'mask']]
        self.connect_bert = False  # 'runn', '##ing' -> 'running' note: this is not exactly de-tokenize!!
        self.connect_bpe = False  # for bpe: 'runn@@', 'ing' -> 'running'
        self.resplit_roberta = False  # resplit roberta
        self.split_zh_char = False  # split zh characters
        self.tmpfile_prefix = "_tmp"  # used for evalers that need to write file!
        # more specific ones
        self.bleu_join_char = ' '  # by default use blank to join it!
        self.sacrebleu_cmd_opt = ''  # if use this, then use specific ones!
        self.sacrebleu_trg_lang = 'en'  # for detokenizer
        self.rouge_use_stemmer = True  # for rouge
        # --

class SeqEvaler:
    def __init__(self, conf: SeqEvalerConf):
        self.conf = conf
        # --
        self._getter_f_gold, _ = inst_obtain_getter_setter(conf.eval_seq_name_gold)
        self._getter_f_pred, _ = inst_obtain_getter_setter(conf.eval_seq_name_pred)
        self._delete_toks_set = set(conf.delete_toks)
        self._tmpfile_prefix = conf.tmpfile_prefix + str(nnutils.ddp_rank()) + str(os.environ.get('EVAL_TMPF_INFIX',''))
        self._zh_reg = re.compile(r"([\u4e00-\u9fa5])")  # [\u4e00-\u9fa5] Chinese range

    def eval(self, gold_insts: List, pred_insts: List, quite=False):
        conf = self.conf
        # get tokens
        gold_seqs = [self._getter_f_gold(z)[0] for z in gold_insts]
        pred_seqs = [self._getter_f_pred(z)[0] for z in pred_insts]
        assert len(gold_seqs) == len(pred_seqs)
        # pre-process: delete certain tokens
        if len(self._delete_toks_set) > 0:
            gold_seqs = [[z1 for z1 in z0 if z1 not in self._delete_toks_set] for z0 in gold_seqs]
            pred_seqs = [[z1 for z1 in z0 if z1 not in self._delete_toks_set] for z0 in pred_seqs]
        # pre-process
        if conf.connect_bert:
            gold_seqs = [self.connect_bert(z) for z in gold_seqs]
            pred_seqs = [self.connect_bert(z) for z in pred_seqs]
        if conf.connect_bpe:
            gold_seqs = [self.connect_bpe(z) for z in gold_seqs]
            pred_seqs = [self.connect_bpe(z) for z in pred_seqs]
        if conf.split_zh_char:
            gold_seqs = [self.split_zh_char(z) for z in gold_seqs]
            pred_seqs = [self.split_zh_char(z) for z in pred_seqs]
        if conf.resplit_roberta:
            gold_seqs = [self.resplit_roberta(z) for z in gold_seqs]
            pred_seqs = [self.resplit_roberta(z) for z in pred_seqs]
        # do eval!
        res = {}
        if not quite:
            zlog("Evaling with SeqEvaler:")
        for eval_name in conf.evals:
            _f = getattr(self, 'eval_' + eval_name)
            one_res = _f(gold_seqs, pred_seqs)
            res[eval_name] = one_res
            if not quite:
                zlog(f"\tRES[{eval_name}] = {one_res}")
        # put one overall res!
        res['res'] = res[conf.evals[0]]['res'] if len(conf.evals) > 0 else 0.
        return res

    # --
    # pre-process

    def connect_bert(self, tokens):
        ret = []
        for t in tokens:
            if t.startswith("##") and len(ret)>0:
                ret[-1] = ret[-1] + t[2:]
            else:
                ret.append(t)
        return ret

    def connect_bpe(self, tokens):
        ret = []
        last_cont = False
        for t in tokens:
            if last_cont:
                ret[-1] = ret[-1][:-2] + t
            else:
                ret.append(t)
            last_cont = t.endswith("@@")
        return ret

    def split_zh_char(self, tokens):
        ret = []
        for t in tokens:
            ones = [z for z in self._zh_reg.split(t) if z]
            ret += ones
        return ret

    def resplit_roberta(self, tokens):
        toks = "".join(tokens).split('Ġ')
        toks = [z for z in toks if len(z)>0]  # remove empty ones
        return toks

    # --
    # evalers

    def eval_cm(self, gold_seqs, pred_seqs):  # complete match!
        assert len(gold_seqs) == len(pred_seqs)
        num_all = len(gold_seqs)
        num_corr = sum([int(a==b) for a,b in zip(gold_seqs, pred_seqs)])
        acc = num_corr / max(1., num_all)
        return {"res": acc, "corr": num_corr, "all": num_all, "details": f"{num_corr}/{num_all}={acc:.4f}"}

    def _parse_bleu(self, res_line: str):
        res_ss = res_line.split("=", 1)[1]
        all_res = [float(z) for z in re.findall(r'[0-9.]+', res_ss)]
        ret = {'res': all_res[0], 'bleu': all_res[0], 'pre4': all_res[1:5], 'BP': all_res[-4], 'ratio': all_res[-3],
               'hyp_len': all_res[-2], 'ref_len': all_res[-1], 'details': res_line}
        return ret

    def eval_bleu(self, gold_seqs, pred_seqs):
        conf = self.conf
        _prefix = self._tmpfile_prefix
        # write temp files
        for seqs, suffix in zip([gold_seqs, pred_seqs], ['.gold', '.pred']):
            with zopen(_prefix+suffix, 'w') as fd:
                for s in seqs:
                    fd.write(conf.bleu_join_char.join(s) + "\n")
        # then run that script
        script_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi-bleu.perl")
        output = system(f"perl {script_name} {_prefix}.gold <{_prefix}.pred", popen=True)
        try:
            res_line = [line.strip() for line in output.split("\n") if line.startswith("BLEU")][0]
            ret = self._parse_bleu(res_line)
        except:
            import traceback
            ret = {'res': 0., 'err': traceback.format_exc(), 'details': output}
        return ret

    def eval_sacrebleu(self, gold_seqs, pred_seqs):
        conf = self.conf
        # --
        from sacremoses import MosesDetokenizer
        md = MosesDetokenizer(lang=conf.sacrebleu_trg_lang)
        gold_ss, pred_ss = [md.detokenize(z) for z in gold_seqs], [md.detokenize(z) for z in pred_seqs]
        if conf.sacrebleu_cmd_opt:
            with zopen(self._tmpfile_prefix+".pred", 'w') as fd:
                for s in pred_ss:
                    fd.write(s+"\n")
            output = system(f"sacrebleu {conf.sacrebleu_cmd_opt} <{self._tmpfile_prefix}.pred", popen=True)
            res_line = [line.strip() for line in output.split("\n") if line.startswith("BLEU")][0]
        else:
            import sacrebleu
            bleu = sacrebleu.corpus_bleu(pred_ss, [gold_ss])
            res_line = str(bleu)
        ret = self._parse_bleu(res_line)
        return ret

    def eval_rouge(self, gold_seqs, pred_seqs):
        conf = self.conf
        # --
        from rouge_score import rouge_scorer, scoring
        metrics = ["rouge1", "rouge2", "rougeL"]
        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=conf.rouge_use_stemmer)
        aggr = scoring.BootstrapAggregator()
        # note: simply join them!
        ii = 0
        for gold_seq, pred_seq in zip(gold_seqs, pred_seqs):
            gold_ss, pred_ss = ' '.join(gold_seq), ' '.join(pred_seq)
            one_res = scorer.score(gold_ss, pred_ss)
            aggr.add_scores(one_res)
            # print(f"{ii}," + ",".join([f"{one_res[m][k]:.6f}" for m in metrics for k in range(3)]))
            ii += 1
        aggr_res = aggr.aggregate()
        ret = {'res': None}
        for score_type, aggregate in sorted(aggr_res.items()):
            ret[score_type+"-R"] = [aggregate.low.recall, aggregate.mid.recall, aggregate.high.recall]
            ret[score_type+"-P"] = [aggregate.low.precision, aggregate.mid.precision, aggregate.high.precision]
            ret[score_type+"-F"] = [aggregate.low.fmeasure, aggregate.mid.fmeasure, aggregate.high.fmeasure]
        ret['res'] = ret["rougeL-F"][1]  # todo(+N): use which one?
        return ret

# --
# b zgen/eval/eval.py:?
