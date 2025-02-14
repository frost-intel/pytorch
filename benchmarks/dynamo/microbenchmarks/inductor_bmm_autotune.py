# Owner(s): ["oncall: cpu inductor"]
import functools
from unittest.mock import patch

import pickle
import torch
import torch._dynamo.config
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch._inductor.select_algorithm as select_algorithm
from torch._dynamo.utils import counters
from torch._inductor.compile_fx import compile_fx
from torch.utils.benchmark import Timer

def time_with_torch_timer(fn, args, kwargs=None, iters=100):
    kwargs = kwargs or {}
    env = {"args": args, "kwargs": kwargs, "fn": fn}
    fn_call = "fn(*args, **kwargs)"

    # Measure end-to-end time
    timer = Timer(stmt=f"{fn_call}", globals=env)
    tt = timer.repeat(repeat=20, number=iters)
    return torch.tensor(tt)

def clone_preserve_strides(x, device=None):
    if not isinstance(x, torch.Tensor):
        return x
    buffer = torch.as_strided(
        x, (x.untyped_storage().size() // x.element_size(),), (1,), 0
    )
    if not device:
        buffer = buffer.clone()
    else:
        buffer = buffer.to(device, copy=True)
    out = torch.as_strided(buffer, x.size(), x.stride(), x.storage_offset())
    return out

def get_compiled_and_ref(
    model,
    nopython=True,
):
    torch._dynamo.reset()
    ref_model = model
    torch.manual_seed(0)
    torch._inductor.metrics.reset()
    called = False

    def compile_fx_wrapper(model_, example_inputs_):
        nonlocal called
        called = True
        return compile_fx(model_, example_inputs_)

    def run(*ex, **kwargs):
        return model(*ex, **kwargs)

    run = torch.compile(run, backend=compile_fx_wrapper, fullgraph=nopython)
    return run, ref_model


def patches(fn):
    def skip_cache(self, choices, name, key, benchmark):
        if benchmark is None:
            return {}
        timings = benchmark(choices)
        return timings

    for patcher in [
        dynamo_config.patch(verbose=True),
        dynamo_config.patch(inline_inbuilt_nn_modules=True),
        inductor_config.patch(
            debug=True,
            max_autotune=True,
            epilogue_fusion=True,
            max_autotune_gemm_backends="CPP,ATEN",
        ),
        patch.object(select_algorithm.AlgorithmSelectorCache, "lookup", skip_cache),
    ]:
        fn = patcher(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        counters.clear()
        torch.manual_seed(12345)
        return fn(*args, **kwargs)

    return wrapped


def arith_intensity(M, N, K):
    return M * N * K / (M * N + N * K + M * K)


@patches
@torch.no_grad
def bench_shape(dtype, shape):
    a_shape, b_shape = shape
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x @ y

    counters.clear()
    B, M, K = a_shape
    N = b_shape[2]
    u = torch.randn(a_shape).to(dtype=dtype)
    v = torch.randn(b_shape).to(dtype=dtype)
    mod = Model().to(dtype=dtype).eval()
    args = (u, v)
    autotuned, base = get_compiled_and_ref(mod, (u, v))
    autotuned(*args)
    base(*args)
    flops = B * M * N * K * 2
    ai = arith_intensity(M, N, K)
    print(flops / 1e9, ai, a_shape, b_shape)
    iters = 1000
    if (flops / 1e9) < 1:
        iters = 5000
    elif (flops / 1e9) > 200:
        iters = 50
    elif (flops / 1e9) > 10:
        iters = 100

    #ind_torch_ms = time_with_torch_timer(base, (u, v), iters=10).median * 1000
    res = time_with_torch_timer(autotuned, (u, v), iters=iters)
    ind_autotune_ms = res.min()
    return ind_autotune_ms, flops, ai, a_shape, b_shape

    #print(dtype, a_shape, "x", b_shape, end="; ")
    #print(ind_torch_ms, ind_autotune_ms, sep="; ")
    

if __name__ == "__main__":
    shapes = (
        # BERT (all)
        ((192, 128, 64), (192, 64, 128)),
        ((192, 128, 128), (192, 128, 64)),
        # BERT_pytorch (all)
        ((47, 128, 64), (47, 64, 128)),
        ((47, 128, 128), (47, 128, 64)),
        ((48, 128, 64), (48, 64, 128)),
        ((48, 128, 128), (48, 128, 64)),
        # hf_GPT2 (all)
        ((12, 1024, 1024), (12, 1024, 64)),
        ((12, 1024, 64), (12, 64, 1024)),
        # hf_Albert (all)
        ((12, 512, 64), (12, 64, 512)),
        ((12, 512, 512), (12, 512, 64)),
        # hf_Reformer (all)
        ((1536, 64, 64), (1536, 64, 128)),
        ((1536, 64, 128), (1536, 128, 64)),
        # dlrm
        #((4, 9, 64), (4, 64, 9)),
        # hf_DistilBert (all)
        ((47, 512, 64), (47, 64, 512)),
        ((47, 512, 512), (47, 512, 64)),
        ((48, 512, 64), (48, 64, 512)),
        ((48, 512, 512), (48, 512, 64)),
        # hf_T5_base (all)
        ((47, 2048, 64), (47, 64, 2048)),
        ((47, 2048, 2048), (47, 2048, 64)),
        ((48, 2048, 64), (48, 64, 2048)),
        ((48, 2048, 2048), (48, 2048, 64)),
        # sam (all)
        #((400, 196, 80), (400, 80, 196)),
        #((14, 5600, 80), (14, 80, 14)),
        ((400, 196, 196), (400, 196, 80)),
        ((16, 1024, 80), (16, 80, 64)),
        #((16, 4096, 4096), (16, 4096, 80)),
        ((1, 4096, 256), (1, 256, 128)),
        ((1, 4, 32), (1, 32, 65536)),
        )
    dtype_tup = (torch.bfloat16, torch.half)
    dtype_tup = (torch.bfloat16,)
    data = {}
    for dtype in dtype_tup:
        for shape in shapes:
            results = bench_shape(dtype, shape)
            data[(dtype, shape)] = results
    pickle.dump(data, open("bmm_autotune_local_change2.pkl", "wb"))
