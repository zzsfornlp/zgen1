#

# training
from zgen.utils import zlog, zwarn, init_everything, Timer, Logger
from ..drive import *

# --
def main(args):
    # conf
    conf: ZOverallConf = init_everything(ZOverallConf(), args)
    # task
    t_center = TaskCenter(conf.tconf)
    # data
    d_center = DataCenter(conf.dconf)
    # build/load vocab: try loading here, and save new built ones!
    _tcf = t_center.conf
    t_center.build_vocabs(d_center)
    t_center.add_preps(d_center)
    # build model
    model = conf.mconf.make_node()
    t_center.build_mods(model)
    # run
    r_center = RunCenter(conf.rconf, model, t_center, d_center)
    if conf.rconf.train_preload_model:
        r_center.load(conf.rconf.train_preload_model)
    model.finish_sr()  # note: build sr after possible loading in training!!
    # --
    # load extra ones?
    load_extra_models(conf.extra_model_specs)
    # --
    r_center.do_train()
    # --
    zlog("The end of Training.")
    Logger.get_singleton_logger().flush_cached_logs()

if __name__ == '__main__':
    import sys
    with Timer(info=f"Training", print_date=True) as et:
        main(sys.argv[1:])

# --
# b zgen/main/train:12
