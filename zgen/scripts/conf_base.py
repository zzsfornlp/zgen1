#

# get basic confs

def args2src(args):
    lines = []
    for a in args:
        if len(a)>2 and a.startswith("'") and a.endswith("'"):
            lines.append(a[1:-1]+"\n")
        else:
            lines.append(a+"\n")
    return "".join(lines)

def strip_quotes(args):
    ret = [a[1:-1] if (len(a)>2 and a.startswith("'") and a.endswith("'")) else a for a in args]
    return ret

def parse_wmt_task(task: str):
    import re
    prefix, src, trg = re.split(r'\d+', task)
    return src, trg

def get_basic_config_bitext(input_dir="", dec_mod="ilm", dec_task="", ilm_mod="", pretrain=""):
    args = []
    # --
    # data
    for wset in ["train0", "dev0", "test0", "test1"]:
        args.extend(f"{wset}.input_dir:{input_dir} {wset}.is_bitext:1".split())
        args.extend(f"{wset}.data_tasks:enc,{dec_mod}".split())
    # specific for train/dev/test
    args.extend(f"train0.batch_size_f:max_len train0.sort_size_f:max_len train0.filter_dump_f:both_len".split())
    # args.extend(f"train0.shuffle_path_times:1".split())  # not doing this ...
    for wset in ["dev0", "test0", "test1"]:
        args.extend(f"{wset}.split_ddp:0 {wset}.output_path:_zout.{wset}.json".split())
    # --
    # model
    # model.enc
    args.extend(f"enc:bert enc.zbert_model:bert-base-cased enc.zbert_extra_setup:baseT enc.zbert_no_pretrain:1".split())
    # model.dec
    args.extend(f"{dec_mod}:yes loss_{dec_mod}:1 {dec_mod}.zbert_model:bert-base-cased {dec_mod}.zbert_extra_setup:baseT {dec_mod}.zbert_no_pretrain:1 {dec_mod}.zbert_add_lmhead:1".split())
    args.extend(f"{dec_mod}.cross_mname:enc {dec_mod}.seq_name:trg".split())
    # about embeddings
    args.extend(f"enc.no_type_embeds:1 {dec_mod}.no_type_embeds:1".split())
    args.extend(f"'share_baset_embeddings:{dec_mod}|enc' {dec_mod}.lmhead_tie_embeds:1".split())
    # extra specific settings for ilm-dec
    if dec_mod == "ilm":
        args.extend(f"rollin_strategy:random0 oracle_strategy:uniform0".split())
    # --
    # training
    args.extend("amp_opt_level:O2 fp16:1 train0.batch_size:6250".split())  # 6250x4=25k
    args.extend("lrate:0.0007 lrate_warmup_uidx:4000 max_uidx:300000 valid_first:0 log_last:1".split())
    # args.extend("save_special_start_cidx:90 save_special_cfreq:2".split())  # save
    args.extend("record_best_start_cidx:10 save_bestn:5".split())  # save
    if dec_mod == 'ilm':  # slightly better for ilm if decay slower
        args.extend("lrate_decrease_alpha:0.4".split())
    # --
    # task-specific
    args.extend(f"{dec_mod}.eval:seq".split())  # eval
    # =====
    if dec_task in ["recons", "reorder", "reconsN"]:  # on wiki data?
        # data
        for wset in ["train0", "dev0", "test0", "test1"]:
            args.extend(f"{wset}.bitext_src_suffix:.txt {wset}.bitext_trg_suffix:.txt".split())
        args.extend(f"train0.input_paths:wiki_00,wiki_01,wiki_02,wiki_03".split())
        args.extend(f"dev0.input_paths:_wiki_dev test0.input_paths:_wiki_dev test1.input_paths:_wiki_test".split())
        if dec_mod == "ilm":
            args.extend(f"{dec_mod}.stat_file:{input_dir}/ALL.cjson".split())
        # task specific
        if dec_task == "recons":  # reconstruction with [cls]
            args.extend("enc.mem_compress_method:idx0 enc.mem_detach:0".split())
        elif dec_task == "reorder":  # reordering
            args.extend("enc.no_posi_embeds:1".split())
        elif dec_task == "reconsN":  # reconstruction with noisy input
            # args.extend("enc.mem_compress_method:idx0 enc.mem_detach:0 enc.noiser:yes".split())
            args.extend("enc.noiser:yes".split())  # no idx0 constraint!
            for wset in ["dev0", "test0", "test1"]:
                args.extend(f"{wset}.bitext_src_suffix:.nsrc {wset}.bitext_trg_suffix:.txt".split())
    elif dec_task == "ro":  # another set of reorder
        # data
        for wset in ["train0", "dev0", "test0", "test1"]:
            args.extend(f"{wset}.bitext_src_suffix:.stok {wset}.bitext_trg_suffix:.stok".split())
        args.extend(f"train0.input_paths:wiki00,wiki01,wiki02,wiki03".split())  # slightly rename!
        args.extend(f"dev0.input_paths:_wiki_dev0 test0.input_paths:_wiki_dev0 test1.input_paths:_wiki_test0".split())
        if dec_mod == "ilm":
            args.extend(f"{dec_mod}.stat_file:{input_dir}/ALL.cjson".split())
        args.extend(f"enc.no_posi_embeds:1 {dec_mod}.do_add_ends:1".split())  # need to add ends!
    # =====
    # note: the above ones are deprecated!
    elif dec_task.startswith("wmt"):  # mt
        # data
        cl_src, cl_trg = parse_wmt_task(dec_task)
        for wset in ["train0", "dev0", "test0", "test1"]:
            args.extend(f"{wset}.is_bitext:1 {wset}.bitext_src_suffix:.{cl_src} {wset}.bitext_trg_suffix:.{cl_trg}".split())
        args.extend(f"train0.input_paths:{dec_task}.train*".split())
        args.extend(f"dev0.input_paths:{dec_task}.dev test0.input_paths:{dec_task}.dev test1.input_paths:{dec_task}.test".split())
        # constrain train length
        args.extend(f"train0.filter_max_length:100".split())
        # voc
        args.extend(f"vocab_load_dir:{input_dir}/voc_{dec_task}/".split())
        if dec_mod == "ilm":
            args.extend(f"{dec_mod}.stat_file:{input_dir}/{dec_task}.cjson".split())
        # model
        args.extend(f"{dec_mod}.do_add_ends:1 {dec_mod}.label_smoothing:0.1".split())
        # eval
        args.extend(f"{dec_mod}.connect_bpe:1".split())
        # special ones
        if cl_trg == "zh":
            args.extend("share_baset_embeddings:".split())  # no share between src and trg for en-zh!
            args.extend(f"{dec_mod}.split_zh_char:1".split())  # eval with splitting zh chars
    # --
    elif dec_task == "comp":  # completion
        # data
        for wset in ["train0", "dev0", "test0", "test1"]:
            args.extend(f"{wset}.bitext_src_suffix:.src {wset}.bitext_trg_suffix:.trg".split())
        args.extend(f"train0.input_paths:train*".split())  # slightly rename!
        args.extend(f"dev0.input_paths:dev test0.input_paths:dev test1.input_paths:test".split())
        if dec_mod == "ilm":
            args.extend(f"{dec_mod}.stat_file:{input_dir}/ALL.cjson".split())
        args.extend(f"enc.no_posi_embeds:0 {dec_mod}.do_add_ends:1".split())  # need to add ends!
        # voc & eval
        args.extend(f"vocab_load_dir:{input_dir}/voc/ {dec_mod}.connect_bpe:1".split())
        # save lastn
        args.extend("lastn_as_bestn:1".split())
        # training
        args.extend("max_uidx:100000".split())
        args.extend("test_beam_size:1 test_do_sample:1 test_sample_topp:0.95".split())
    elif dec_task == "ro103":  # reorder with wiki103
        # data
        for wset in ["train0", "dev0", "test0", "test1"]:
            args.extend(f"{wset}.bitext_src_suffix:.stok {wset}.bitext_trg_suffix:.stok".split())
        args.extend(f"train0.input_paths:train*".split())  # slightly rename!
        args.extend(f"dev0.input_paths:dev test0.input_paths:dev test1.input_paths:test".split())
        if dec_mod == "ilm":
            args.extend(f"{dec_mod}.stat_file:{input_dir}/ALL.cjson".split())
        args.extend(f"enc.no_posi_embeds:1 {dec_mod}.do_add_ends:1".split())  # need to add ends!
        # voc & eval
        args.extend(f"vocab_load_dir:{input_dir}/voc/ {dec_mod}.connect_bpe:1".split())
        # training
        args.extend("max_uidx:100000".split())
    elif dec_task == "xsum":  # xsum
        # data
        for wset in ["train0", "dev0", "test0", "test1"]:
            args.extend(f"{wset}.bitext_src_suffix:.stokS0 {wset}.bitext_trg_suffix:.stokT {wset}.batch_size_f:1 {wset}.batch_size:10".split())
        args.extend(f"train0.input_paths:train*".split())  # slightly rename!
        args.extend(f"dev0.input_paths:dev test0.input_paths:dev test1.input_paths:test".split())
        if dec_mod == "ilm":
            args.extend(f"{dec_mod}.stat_file:{input_dir}/ALL.cjson".split())
        args.extend(f"enc.no_posi_embeds:0 {dec_mod}.do_add_ends:1".split())  # need to add ends!
        # voc & eval
        args.extend(f"vocab_load_dir:{input_dir}/voc/ {dec_mod}.connect_bpe:1".split())
        # small dev
        args.extend("dev0.take_first:500".split())
        # training
        args.extend("max_uidx:100000 lrate:0.0004 train0.batch_size:32".split())
    # --
    # take first several for dev to save time
    # args.extend("dev0.take_first:500".split())
    # --
    # extra ones for ilm
    if dec_task == 'comp':  # note: no penalty since no easy way to decide
        _baseC = ["""'train_end_test_conditions:[f"self.model.Milm.set_test_confs({z0},0.)" for z0 in [0.,0.95]]'""", "ilm.test_noi_penalty:-0."]
    else:
        _baseC = ["""'train_end_test_conditions:[f"self.model.Milm.set_test_noi_penalty({-z/2.})" for z in range(11)]'"""]
    _base_m0 = """ ilm.train_onestep_model0:1 ilm.incr_embed:no ilm.incr_layers:0 ilm.incr_grow_hid:0 oracle_strategy:bt0 ilm.test_noi_penalty:-2. """  # generally -2 seems good for m0
    _base_m1 = """ ilm.no_posi_embeds:1 ilm.incr_embed:yes ilm.incr_layers:6 ilm.incr_grow_hid:0 rollin_strategy:bt0 oracle_strategy:bt0 ilm.test_noi_penalty:-0. """
    if ilm_mod == "m0":
        args.extend(_base_m0.split() + _baseC)
    elif ilm_mod == "m1":
        args.extend(_base_m1.split() + _baseC)
    if dec_task == 'comp' and dec_mod == 'slm':
        args.extend(["""'train_end_test_conditions:[f"self.model.Mslm.set_test_confs({z0},0.)" for z0 in [0.,0.95]]'"""])
    # =====
    # finally special one for pretrained ones
    if pretrain:
        args.extend(f"enc.zbert_model:{pretrain} enc.zbert_extra_setup: enc.zbert_no_pretrain:0".split())
        args.extend(f"{dec_mod}.zbert_model:{pretrain} {dec_mod}.zbert_extra_setup: {dec_mod}.zbert_no_pretrain:0".split())
        args.extend(f"enc.no_type_embeds:0 {dec_mod}.no_type_embeds:0".split())
        args.extend("vocab_load_dir:".split())  # directly using pretrained tokenizer/vocab!
        args.extend("lrate:0.0001".split())  # no need that large lrate ...
        if 'roberta' in pretrain or 'bart' in pretrain:
            args.extend(f"{dec_mod}.resplit_roberta:1".split())
            # todo(+N): for simplicity use this special one; and currently PAU is not utilized!!
            args.extend(f"'enc.noi_token_name:<mask>'".split())
            args.extend(f"'enc.pau_token_name:<mask>'".split())
            args.extend(f"'{dec_mod}.noi_token_name:<mask>'".split())
            args.extend(f"'{dec_mod}.pau_token_name:<mask>'".split())
    # --
    return args
