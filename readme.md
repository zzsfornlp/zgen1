### For Insertion Generation with FPE

Hi, this describes our implementation for our paper: "Towards More Efficient Insertion Transformer with Fractional Positional Encoding".

Please refer to the paper for more details: [[paper]](TODO) [[bib]](TODO)

### Setup

Clone this repo:

	git clone https://github.com/zzsfornlp/zgen1/ src

Please refer to the [`scripts/setup_env.sh`](./scripts/setup_env.sh) for details on setting up environments.

Before running anything, make sure to export the src directory to your `$PYTHONPATH`:

    export PYTHONPATH=/your/path/to/src

### Data Preparation

Data and vocabs should be prepared before running. Here is an example for preparing MT data (Distilled WMT14-en-de), we simply use the prepared ones from [fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/nonautoregressive_translation/README.md).

Please refer to the script [`scripts/prepare-wmt14en2dev2.sh`](./scripts/prepare-wmt14en2dev2.sh) for more details.

### Training

Assuming the data is prepared at the directory of `!!!!!MYDATA!!!!!` (replace this in the command with the real data path!), use the following commands to train an FPE-based Insertion Model (here we are assuming using 2 GPUs, thus leading to a batch size of 12500*2=25K, please adjust the batch size according to the number of GPUs that you are using):

    mkdir run
    cd run
    python -m zgen.main.run_ddp zgen.main.train log_file:_log log_last:1 conf_output:_confR  'train_end_test_conditions:[f"self.model.Milm.set_test_noi_penalty({-z/2.})" for z in range(10)]'  ilm.no_posi_embeds:1 ilm.incr_embed:yes ilm.incr_layers:6 ilm.incr_grow_hid:0 rollin_strategy:bt0 oracle_strategy:bt0 ilm.test_noi_penalty:-0.5 'conf_sbase:{"input_dir":"!!!!!MYDATA!!!!!","dec_mod":"ilm","dec_task":"wmt14en2de"}' max_uidx:300000 ilm.test_noi_penalty:-1. train0.batch_size:12500 train0.bitext_src_suffix:.enD train0.bitext_trg_suffix:.deD lrate_decrease_alpha:0.4

### Speed Testing

For testing with different batch sizes (`$BSIZE`), testing the speed with (here, `test_certain_batches` specifies how many batches to run for speed testing with repeated sampling, deleting this option will make it ordinary testing):

    python3 -m zgen.main.test _conf nn.dist_world_size:1 fp16:$FP16 model_load_name:zmodel.bestn.m test0.input_dir:${MYDATA} test1.input_paths: test0.input_paths:wmt14en2de.test vocab_load_dir:${MYDATA}/voc_wmt14en2de/ test0.batch_size:$BSIZE test_certain_batches:100
