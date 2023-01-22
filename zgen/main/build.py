#

# build vocabs for tasks
from zgen.utils import zlog, zwarn, init_everything, Timer
from ..drive import *

# --
def main(args):
    # conf
    conf: ZOverallConf = init_everything(ZOverallConf(), args)
    # task
    t_center = TaskCenter(conf.tconf)
    # data
    d_center = DataCenter(conf.dconf)
    # task-center prepare
    t_center.build_vocabs(d_center)  # build
    t_center.add_preps(d_center)  # add prep
    # loop train-data for cache
    # counts = [[], []]
    # for ii in range(2):
    #     for dataset in d_center.get_datasets(wset="train"):
    #         counts[ii].append(0)
    #         for batch in dataset.yield_batches(loop=False):
    #             counts[ii][-1] += len(batch)
    # assert counts[0] == counts[1]
    # --
    zlog("The end of Building.")

if __name__ == '__main__':
    import sys
    with Timer(info=f"Building", print_date=True) as et:
        main(sys.argv[1:])
