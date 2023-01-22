#

# convert vocab

from collections import Counter
from zgen.utils import zlog, zopen, default_pickle_serializer
from zgen.data import SimpleVocab

def main(voc_input, voc_output):
    cc = Counter()
    all_words = [f'[unused{ii}]' for ii in range(10)]
    with zopen(voc_input) as fd:
        for line in fd:
            ww, _ = line.split()
            all_words.append(ww)
            cc['words'] += 1
    # --
    voc = SimpleVocab.build_by_static(
        all_words, pre_list=("pad", "unk", "eos", "bos", "mask", "noi", "pau", "s1", "s2", "s3"), post_list=())
    # write
    default_pickle_serializer.to_file(voc, voc_output)
    zlog(f"Build {voc} and write to {voc_output}")
    # --

# PYTHONPATH=?? python3 convert_vocab.py IN OUT
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
