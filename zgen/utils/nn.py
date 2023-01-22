#

# related with NN

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import math
from .conf import Conf
from .log import zlog, zwarn
from .file import WithWrapper

# NN conf
class NNConf(Conf):
    def __init__(self):
        self.random_seed = 9347
        self.random_cuda_seed = 9349
        self.num_threads = 4  # maximum NUM_THREADS if using cpu
        self.device = -1  # -1: cpu, [0,): gpu
        self.fp16 = False  # use fp16 as default?
        self.force_model_dtype = False  # force model to this default float type! (note: usually no need for this!)
        # dist
        self.dist_backend = "nccl"
        self.dist_rank = 0
        self.dist_world_size = 1  # really activate if >1
        self.dist_find_unused_parameters = False
        # amp (by apex)
        self.amp_opt_level = ""  # O0/O1/O2/O3
        # amp (by torch)
        self.use_torch_amp = False
        # --

# singleton
_global_tr_conf = NNConf()
def get_global_conf():
    return _global_tr_conf
def set_gloabl_conf(conf: NNConf):
    global _global_tr_conf
    _global_tr_conf = conf
# global resources
Expr = torch.Tensor
Module = torch.nn.Module
CPU_DEVICE = torch.device("cpu")
DEFAULT_DEVICE = CPU_DEVICE
Function = torch.autograd.Function
DEFAULT_FLOAT = torch.float32
DEFAULT_INT = torch.long
TORCH2NP = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    None: None,
}
# --

def init(conf: NNConf = None):
    if conf is None:
        conf = get_global_conf()
    else:
        set_gloabl_conf(conf)
    # --
    torch.set_num_threads(conf.num_threads)
    torch.manual_seed(conf.random_seed)
    torch.cuda.manual_seed(conf.random_cuda_seed)
    if conf.device >= 0:
        global DEFAULT_DEVICE
        DEFAULT_DEVICE = torch.device(f"cuda:{conf.device}")
    if conf.dist_world_size > 1:
        import os
        _env = os.environ
        if 'MASTER_ADDR' in _env and 'MASTER_PORT' in _env:
            # initialize the process group
            torch.cuda.set_device(conf.dist_rank)
            master_url = f"tcp://{_env['MASTER_ADDR']}:{_env['MASTER_PORT']}"
            dist.init_process_group(conf.dist_backend, init_method=master_url, rank=conf.dist_rank, world_size=conf.dist_world_size)
            zlog(f"Init dist with [{master_url}] {conf.dist_rank}/{conf.dist_world_size}.")
        else:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12365'
            # initialize the process group
            torch.cuda.set_device(conf.dist_rank)
            dist.init_process_group(conf.dist_backend, rank=conf.dist_rank, world_size=conf.dist_world_size)
            zlog(f"Init dist with {conf.dist_rank}/{conf.dist_world_size}.")
    if conf.fp16:
        global DEFAULT_FLOAT
        DEFAULT_FLOAT = torch.float16
    # --

# --
# ddp related
def use_ddp():
    return get_global_conf().dist_world_size > 1

def ddp_world_size():
    return get_global_conf().dist_world_size

def ddp_rank():
    return get_global_conf().dist_rank

def is_main_process():
    return get_global_conf().dist_rank <= 0  # rank0 is the main one!!

# --
# from fairseq
class ModuleProxyWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        assert hasattr(module, "module"), \
            "ModuleProxyWrapper expects input to wrap another module"
        self.module = module

    def __getattr__(self, name):
        """Forward missing attributes to twice-wrapped module."""
        try:
            # defer to nn.Module's logic
            return super().__getattr__(name)
        except AttributeError:
            try:
                # forward to the once-wrapped module
                return getattr(self.module, name)
            except AttributeError:
                # forward to the twice-wrapped module
                return getattr(self.module.module, name)

    def state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        return self.module.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        return self.module.module.load_state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
# --

def wrap_ddp_model(model):
    conf = get_global_conf()
    if conf.amp_opt_level:
        from apex.parallel import DistributedDataParallel as DDP
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP0
        DDP = lambda model: DDP0(model, device_ids=[conf.device])
    # --
    m1 = DDP(model)
    m2 = ModuleProxyWrapper(m1)
    return m2
# --

# --
# pytorch's amp
def autocast_env(enabled=None):
    if enabled is None:
        conf = get_global_conf()
        enabled = conf.use_torch_amp
    if enabled:
        try:
            ret = torch.cuda.amp.autocast(enabled=enabled)
            return ret
        except:
            zwarn(f"There are no autocast in current version: {torch.__version__}")
    return WithWrapper()

# dummy scaler that does nothing!!
class DummyGradScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        return

def get_grad_scaler(enabled=None):
    if enabled is None:
        conf = get_global_conf()
        enabled = conf.use_torch_amp
    if enabled:
        try:
            ret = torch.cuda.amp.GradScaler(enabled=enabled)
            return ret
        except:
            zwarn(f"There are no GradScaler in current version: {torch.__version__}")
    return DummyGradScaler()
# --

def refresh():
    # not doing this, since it makes things slower
    # torch.cuda.empty_cache()
    pass

# todo(note): default init
init()
# --

def save_model(model: Module, path: str):
    d = model.state_dict()
    torch.save(d, path)

def load_model(model: Module, path: str, strict=True, map_location=None):
    if map_location is None:
        map_location = DEFAULT_DEVICE
    d = torch.load(path, map_location=map_location)
    model.load_state_dict(d, strict=strict)

# --
# prepare data

def is_expr(v):
    return isinstance(v, Expr)
is_tensor = is_expr

def is_zero_shape(t, dim=None):
    shapes = list(t.shape)
    return any(s==0 for s in shapes) if (dim is None) else shapes[dim]==0

# inputs with dtype
def input_tensor(inputs, dtype=None, device=None):
    if is_expr(inputs):
        return inputs
    dtype = (DEFAULT_FLOAT if dtype is None else dtype)
    device = (DEFAULT_DEVICE if device is None else device)
    return torch.tensor(inputs, dtype=dtype, device=device)

# (inputs: python data type) -> FloatTensor
def input_real(inputs, device=None):
    return input_tensor(inputs, dtype=DEFAULT_FLOAT, device=device)

# (inputs: python data type of indexes) -> LongTensor
def input_idx(inputs, device=None):
    return input_tensor(inputs, dtype=DEFAULT_INT, device=device)

# arange
def arange(*args, dtype=None, device=None):
    dtype = (DEFAULT_INT if dtype is None else dtype)
    device = (DEFAULT_DEVICE if device is None else device)
    return torch.arange(*args, dtype=dtype, device=device)

# (shape: ..., value: float) -> FloatTensor
def constants(shape, value=0., dtype=None, device=None):
    dtype = (DEFAULT_FLOAT if dtype is None else dtype)
    device = (DEFAULT_DEVICE if device is None else device)
    return torch.full(shape, value, dtype=dtype, device=device)

zeros = lambda shape, **kwargs: constants(shape, **kwargs)

# return 2D eye matrix
def eye(n, dtype=None, device=None):
    dtype = (DEFAULT_FLOAT if dtype is None else dtype)
    device = (DEFAULT_DEVICE if device is None else device)
    return torch.eye(n, dtype=dtype, device=device)

# (shape: ..., p: rate of 1., mul: multiply) -> Tensor
def random_bernoulli(shape, p: float, mul: float, dtype=None, device=None):
    dtype = (DEFAULT_FLOAT if dtype is None else dtype)
    device = (DEFAULT_DEVICE if device is None else device)
    x = torch.full(shape, p, dtype=dtype, device=device)
    r = torch.bernoulli(x) * mul
    return r

# [0,1)
def rand(shape, dtype=None, device=None):
    dtype = (DEFAULT_FLOAT if dtype is None else dtype)
    device = (DEFAULT_DEVICE if device is None else device)
    return torch.rand(shape, dtype=dtype, device=device)

# activation functions
ACT2FN = {"gelu": F.gelu, "relu": F.relu}

# --
# batch 2d tensor
def go_batch_2d(inputs, pad_val, max_len=None, dtype=None, return_tensor=True):
    bs = len(inputs)
    if max_len is None:
        max_len = 1 if bs == 0 else max(len(z) for z in inputs)
    if dtype is None:  # guess dtype
        if isinstance(pad_val, int): dtype = DEFAULT_INT
        elif isinstance(pad_val, float): dtype = DEFAULT_FLOAT
    arr = np.full([bs, max_len], pad_val, dtype=TORCH2NP[dtype])
    arr_mask = np.full([bs, max_len], 0., dtype=np.float32)
    for ii, vv in enumerate(inputs):
        if len(vv) <= max_len:  # normal
            arr[ii, :len(vv)] = vv
            arr_mask[ii, :len(vv)] = 1.
        else:  # truncate
            arr[ii, :max_len] = vv[:max_len]
            arr_mask[ii, :max_len] = 1.
    if return_tensor:
        return input_tensor(arr, dtype=dtype), input_tensor(arr_mask, dtype=DEFAULT_FLOAT)
    else:
        return arr, arr_mask

# no grad env
def no_grad_env(no_grad=True):
    if no_grad:
        return torch.autograd.no_grad()
    else:
        return WithWrapper(None, None, None)

def loss_nll(score_expr, gold_idxes, label_smoothing=0.):
    gold_idxes_t = input_idx(gold_idxes)
    # no average or sum-reduce for the output
    # output = F.nll_loss(F.log_softmax(score_expr, dim=-1), gold_idxes_t, size_average=False, reduce=False)
    # log_softmax_score = F.log_softmax(score_expr, dim=-1)
    # if label_smoothing > 0.:
    #     N = score_expr.size(-1) - 1.
    #     weight = score_expr.new_ones(score_expr.size()) * (label_smoothing / N)
    #     weight.scatter_(-1, gold_idxes.unsqueeze(-1), (1. - label_smoothing))
    #     ret = - (weight * log_softmax_score).sum(dim=-1)  # [*, C]
    #     # # note: substract baseline
    #     # ret += ((1-label_smoothing) * math.log(1-label_smoothing) + label_smoothing * math.log(label_smoothing/N + 1e-10))
    # else:
    #     ret = - log_softmax_score.gather(-1, gold_idxes_t.unsqueeze(-1)).squeeze(-1)  # [*, C]
    # --
    # keep it simple!
    nll_t = - score_expr.log_softmax(dim=-1)  # [*, C, V]
    ret_t = nll_t.gather(-1, gold_idxes_t.unsqueeze(-1)).squeeze(-1)  # [*, C]
    if label_smoothing > 0.:
        ret_t = (1.-label_smoothing) * ret_t + label_smoothing * nll_t.mean(-1)  # [*, C]
    return ret_t

def get_value(t):
    return t.detach().cpu().numpy()

def set_value(t, val, resize=False):
    with torch.autograd.no_grad():  # avoid recording grad_fn
        if not resize:
            assert t.shape == val.shape
        # if resize:  # if need to resize
        #     src_shape, trg_shape = get_shape(t), get_shape(val)
        #     if src_shape != trg_shape:
        #         t.resize_(trg_shape)
        t.set_(input_real(val))

# --
# more complex helpers & special routines

# todo(note): for mask->idx: 1) argsort, 2) pad 1s + nonzero, 3) loop;
# the inputs should be 1. or 0. (float); [*, L, *] -> [*, max-count, *]
def mask2idx_v1(mask_f, dim=-1, pad=0):
    mask_shape = mask_f.shape  # [*, L, *]
    # --
    # judge zero-shape
    if any(z==0 for z in mask_shape):
        _shape = list(mask_shape)
        _shape[dim] = 1
        zz = torch.zeros(_shape)  # [*, 1, *], put an one here!
        return zz.to(DEFAULT_INT), zz.to(DEFAULT_FLOAT)
    # --
    _flt = torch.float32
    mask_f = mask_f.to(_flt)  # [*, L, *]
    max_len = mask_shape[dim]  # L
    # prepare sorting
    to_add = (max_len - 1 - arange(max_len).to(_flt)) / max_len  # [L], <1, put lower idxes larger
    to_expand_dim = (-dim-1) if dim<0 else (len(mask_shape)-1-dim)
    to_add = to_add.view([max_len] + [1]*to_expand_dim)  # [L, *]
    to_sort = mask_f + to_add  # [*, L, *]
    # sort it
    sort_vals, sort_idxes = to_sort.sort(dim=dim, descending=True)  # [*, L, *], descending!!
    sort_valids = (sort_vals>=1.).to(DEFAULT_FLOAT)  # [*, L, *]
    # shrink it!
    max_count = max(1, int(mask_f.sum(dim=dim).max().item()))  # M
    res_idxes, res_mask = sort_idxes.narrow(dim, 0, max_count), sort_valids.narrow(dim, 0, max_count)  # [*, M, *]
    res_idxes[res_mask<=0.] = pad  # padding
    return res_idxes, res_mask

def mask2idx_v2(mask_f, dim=-1, pad=0):
    mask_shape = mask_f.shape  # [*, L, *]
    # --
    # judge zero-shape
    if any(z==0 for z in mask_shape):
        _shape = list(mask_shape)
        _shape[dim] = 1
        zz = torch.zeros(_shape)  # [*, 1, *], put an one here!
        return zz.to(DEFAULT_INT), zz.to(DEFAULT_FLOAT)
    # --
    _flt = torch.float32
    mask_f = mask_f.to(_flt)  # [*, L, *]
    # get max counts
    counts = mask_f.sum(dim=dim, keepdims=True)  # [*, 1, *]
    max_count = max(1, int(counts.max().item()))  # M
    padding_counts = max_count - counts  # [*, 1, *]
    max_padding_count = int(padding_counts.max().item())  # int, the max count of padding
    # pad and concat
    _arange_idx = arange(max_padding_count)  # [mp]
    to_expand_dim = (-dim-1) if dim<0 else (len(mask_shape)-1-dim)
    pad_t = (_arange_idx.view([max_padding_count]+[1]*to_expand_dim) < padding_counts).float()  # [*, mp, *]
    concat_t = torch.cat([mask_f, pad_t], dim)  # [*, L+mp, *]
    # nonzero and extract
    final_shape = list(mask_shape)
    final_shape[dim] = max_count
    ret_idxes = concat_t.nonzero(as_tuple=False)[:, dim].reshape(final_shape)  # [*, M, *]
    # get valid mask and set pad for invalid ones
    max_len = mask_shape[dim]  # L
    valid_mask = (ret_idxes < max_len).to(DEFAULT_FLOAT)  # [*, M, *]
    ret_idxes[valid_mask<=0.] = pad
    return ret_idxes, valid_mask

def mask2idx_v3(mask_f, dim=-1, pad=0):
    mask_shape = mask_f.shape  # [*, L, *]
    # --
    # judge zero-shape
    if any(z==0 for z in mask_shape):
        _shape = list(mask_shape)
        _shape[dim] = 1
        zz = torch.zeros(_shape)  # [*, 1, *], put an one here!
        return zz.to(DEFAULT_INT), zz.to(DEFAULT_FLOAT)
    # --
    _flt = torch.float32
    mask_f = mask_f.to(_flt)  # [*, L, *]
    max_len = mask_shape[dim]  # L
    # loop
    res0_idxes = torch.full(mask_shape, pad, dtype=DEFAULT_INT)  # [*, L, *]
    slice_shape = list(mask_shape)
    slice_shape[dim] = 1  # [*, 1, *]
    cur_idx = torch.zeros(slice_shape, dtype=DEFAULT_INT)  # [*, 1, *]
    for one_idx, one_slice in enumerate(mask_f.split(1, dim=dim)):
        one_hit = (one_slice > 0).to(DEFAULT_INT)  # [*, 1, *]
        res0_idxes.scatter_(dim, cur_idx, one_idx)
        cur_idx += one_hit  # if hit, move one space more!
    # --
    counts = cur_idx  # [*, 1, *]
    _arange_idx = arange(max_len)  # [L]
    to_expand_dim = (-dim-1) if dim<0 else (len(mask_shape)-1-dim)
    res0_mask = (_arange_idx.view([max_len]+[1]*to_expand_dim) < counts).to(DEFAULT_FLOAT)
    # shrink it!
    max_count = max(1, int(counts.max().item()))  # M
    res_idxes, res_mask = res0_idxes.narrow(dim, 0, max_count), res0_mask.narrow(dim, 0, max_count)  # [*, M, *]
    res_idxes[res_mask<=0.] = pad  # padding
    return res_idxes, res_mask

# --
mask2idx = mask2idx_v2  # choose v2 since it is the fastest!!
# --

# --
def _check_mask2idx():
    import time
    CC = 100
    SHAPE = [1024, 200]
    t1, t2, t3 = 0, 0, 0
    for i in range(CC):
        m = (torch.rand(SHAPE) > 0.5).float()
        c1 = time.time()
        x1, x2 = mask2idx_v1(m)
        c2 = time.time()
        y1, y2 = mask2idx_v2(m)
        c3 = time.time()
        z1, z2 = mask2idx_v3(m)
        c4 = time.time()
        t1 += c2 - c1
        t2 += c3 - c2
        t3 += c4 - c3
        print((x1-y1).abs().sum(), (x2-y2).abs().sum())
        print((x1-z1).abs().sum(), (x2-z2).abs().sum())
        assert torch.allclose(x1, y1) and torch.allclose(x2, y2)
        assert torch.allclose(x1, z1) and torch.allclose(x2, z2)
    print(f"{t1} vs {t2} vs {t3}")
    # => 0.7464661598205566 vs 0.30445027351379395 vs 0.49973011016845703
# --

# sampling 1 with gumble
# argmax(logprob + -log(-log(Unif[0,1]))), return (val, idx)
def category_sample(logprob, dim=-1, keepdim=True, eps=1e-10, top_k=0, top_p=0.0):
    # --
    # note: from
    filter_value = float('-inf')
    if top_k > 0:
        logprob = logprob.clone()  # clone it to modify inplace
        top_k = min(top_k, logprob.size(dim))  # Safety check
        indices_to_remove = logprob < (logprob.topk(top_k, dim=dim)[0].narrow(dim, -1, 1))
        logprob[indices_to_remove] = filter_value
    if top_p > 0.:
        logprob = logprob.clone()  # clone it to modify inplace
        sorted_logits, sorted_indices = logprob.sort(dim=dim, descending=True)
        cumulative_probs = sorted_logits.softmax(dim).cumsum(dim)
        idx_boundary = (cumulative_probs <= top_p).long().sum(dim, keepdims=True)  # [..., 1, ...]
        idx_boundary.clamp_(max=logprob.size(dim)-1)
        value_boundary = sorted_logits.gather(dim, idx_boundary)
        logprob[logprob<value_boundary] = filter_value
    # --
    G = torch.rand(logprob.shape, dtype=torch.float32, device=DEFAULT_DEVICE)
    X = logprob - (-(G+eps).log() + eps).log()
    V, I = X.max(dim, keepdim=keepdim)
    return None, I

# select topk ones
# [*, D, *], [*, 1, *], [*, D, *]
def select_topk(score_t, topk_t, mask_t=None, dim=-1, noise=0.):
    # prepare K
    if isinstance(topk_t, int):
        K = topk_t
        tmp_shape = list(score_t.shape)
        tmp_shape[dim] = 1  # set it as 1
        topk_t = constants(tmp_shape, K, dtype=DEFAULT_INT)  # [*, 1, *]
    else:
        K = topk_t.max().item()
    exact_rank_t = topk_t - 1  # [bsize, 1]
    exact_rank_t.clamp_(min=0, max=K-1)  # make it in valid range!
    # mask values
    if mask_t is not None:
        _extra_score = torch.zeros_like(score_t)
        _extra_score[mask_t<=0.] = float('-inf')
        score_t = score_t + _extra_score
    # add some small noise to break tie
    if noise > 0:
        _extra_noise = torch.rand_like(score_t) * noise
        score_t = score_t + _extra_noise
    # topk
    topk_vals, _ = score_t.topk(K, dim, largest=True, sorted=True)  # [*, K, *]
    # gather score
    sel_thresh = topk_vals.gather(dim, exact_rank_t)  # [*, 1, *]
    # get topk_mask (if K is 0, then select nothing!)
    topk_mask = ((score_t >= sel_thresh) & (topk_t > 0)).to(DEFAULT_FLOAT)  # [*, D, *]
    if mask_t is not None:
        topk_mask *= mask_t
    return topk_mask

# --
# cache aranges
_arange_caches = {}
def get_arange_t(bsize: int, unsqueeze_num: int):
    _key = (bsize, unsqueeze_num)
    ret = _arange_caches.get(_key, None)
    if ret is None:
        ret = arange(bsize, dtype=DEFAULT_INT)  # [*]
        for i in range(unsqueeze_num):
            ret = ret.unsqueeze(-1)
        _arange_caches[_key] = ret
    return ret
