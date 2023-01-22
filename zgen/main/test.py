#

# testing
from zgen.utils import zlog, zwarn, init_everything, Timer
from zgen.utils import nn as nnutils
from ..drive import *

# --
def main(args):
    # conf
    conf: ZOverallConf = init_everything(ZOverallConf(), args)
    # task
    t_center = TaskCenter(conf.tconf)
    # data
    d_center = DataCenter(conf.dconf, specified_wset=["test"])
    # load vocab
    t_center.load_vocabs(t_center.conf.vocab_load_dir)
    t_center.add_preps(d_center)
    # build model
    model = conf.mconf.make_node()
    t_center.build_mods(model)
    model.finish_sr()  # note: build sr before possible loading in testing!!
    model.to(nnutils.DEFAULT_FLOAT)  # move to default float!!
    # run
    r_center = RunCenter(conf.rconf, model, t_center, d_center)
    if conf.rconf.model_load_name != "":
        r_center.load(conf.rconf.model_load_name)
    else:
        zwarn("No model to load, Debugging mode??")
    # --
    # load extra ones?
    load_extra_models(conf.extra_model_specs)
    # --
    if r_center.conf.test_do_iter:
        r_center.do_iter_test()
    else:
        res = r_center.do_test()
        zlog(f"zzzztestfinal: {res}")
    # --
    zlog("The end of Testing.")

if __name__ == '__main__':
    import sys
    with Timer(info=f"Testing", print_date=True) as et:
        main(sys.argv[1:])
