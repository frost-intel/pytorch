# mypy: allow-untyped-defs
from typing import Any, Callable, cast, List, Optional, Union

import torch
import torch.utils
from .. import ir, lowering as L

from ..kernel.mm_common import mm_args
from ..select_algorithm import DataProcessorTemplateWrapper
from ..utils import cache_on_self, has_free_symbols, parallel_num_threads
from ..virtualized import V
from .cpp_micro_gemm import create_micro_gemm
from .cpp_template import CppTemplate

from .cpp_template_kernel import CppTemplateKernel
from .cpp_gemm_template import GEMM_TEMPLATE
from .cpp_utils import DTYPE_TO_CPP, GemmBlocking

MM_TEMPLATES = r"""
void one_thread_mm(
    const {{micro_gemm.get_common_options()['input_t']}}* X,
    const {{micro_gemm.get_common_options()['input_t']}}* W,
    {{micro_gemm.get_common_options()['input_t']}}* Y
    {%- if is_dynamic_M %},
    const int64_t {{kernel.size(GemmOut, -2, unwrapped=True)}}
    {% endif %}
)
{
    {{kernel.maybe_codegen_profile()}}
    constexpr int64_t N = {{kernel.size(GemmOut, -1)}};
    constexpr int64_t K = {{kernel.size(X, -1)}};
    constexpr int64_t M0 = {{micro_gemm.register_blocking.block_m}};
    constexpr int64_t N0 = {{micro_gemm.register_blocking.block_n}};
    constexpr int64_t K0 = {{micro_gemm.register_blocking.block_k}};
    constexpr int64_t N0_blocks = (N + N0 - 1) / N0;
    constexpr int64_t K0_blocks = (K + K0 - 1) / K0;

    static_assert(N % N0 == 0, "N dimension must be multiple of N0");

    // TODO(jgong5): improve cache blocking with CPU info (Mc, Kc)
    {%- if is_dynamic_M %}
    const int64_t M = {{kernel.size(GemmOut, -2)}};
    const int64_t M0_blocks = (M + M0 - 1) / M0;
    const auto Mt_blocks = M0_blocks;
    const auto Nt_blocks = N0_blocks;
    const auto Kt_blocks = K0_blocks;
    const int64_t Mc_blocks = Mt_blocks;
    const int64_t Kc_blocks = Kt_blocks;
    {%- else %}
    constexpr int64_t M = {{kernel.size(GemmOut, -2)}};
    constexpr int64_t M0_blocks = (M + M0 - 1) / M0;
    constexpr int64_t Mt_blocks = {{template.thread_blocking().block_m}};
    constexpr int64_t Nt_blocks = {{template.thread_blocking().block_n}};
    constexpr int64_t Kt_blocks = {{template.thread_blocking().block_k}};
    constexpr int64_t Mc_blocks = {{template.cache_blocking().block_m}};
    constexpr int64_t Kc_blocks = {{template.cache_blocking().block_k}};
    {%- endif %}

    int64_t m_block_start = 0;
    int64_t m_block_end = M0_blocks;
    int64_t n_block_start = 0;
    int64_t n_block_end = N0_blocks;
    int64_t k_block_start = 0;
    int64_t k_block_end = K0_blocks;
    for (int64_t mc = m_block_start; mc < m_block_end; mc += Mc_blocks) {
        const int64_t m_start = mc * M0;
        const int64_t m_end = std::min((mc + Mc_blocks) * M0, M);
        const int64_t m_size = m_end - m_start;
        {%- if use_local_acc %}
        {{ kernel.define_buffer(acc_buf_name, [1, "m_end - m_start", "N0"]) }}
        {%- endif %}
        for (int64_t nc = n_block_start; nc < n_block_end; ++nc) {
            const int64_t n_start = nc * N0;
            const int64_t n_size = N0;
            {%- if use_local_acc %}
            {%- set acc = kernel.local_buffers[acc_buf_name] %}
            {%- else %}
            {%- set acc = kernel.slice_nd(GemmOut, [(0, 1), ("m_start", "m_end"), ("n_start", "n_start + N0")]) %}
            {%- endif %}
            for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                int64_t k_start = kc * K0;
                int64_t k_end = std::min((kc + Kc_blocks) * K0, K);
                {%- set tile_X = kernel.slice_nd(X, [(0, 1), ("m_start", "m_end"), ("k_start", "k_end")]) %}
                {%- set tile_W = kernel.slice_nd(W, [(0, 1), ("k_start", "k_end"), ("n_start", "n_start + n_size")]) %}
                if (kc == k_block_start) {
                    {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc, accum=False, is_bmm=is_bmm)|indent(24, false) }}
                } else {
                    {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc, accum=True, is_bmm=is_bmm)|indent(24, false) }}
                }
            }
            {%- set tile_Y = kernel.slice_nd(Y_2d, [(0, 1), ("m_start", "m_end"), ("n_start", "n_start + N0")]) %}
            {{ kernel.store_output(
                tile_Y, acc, GemmOut, epilogue_nodes, offsets=(0, "m_start", "n_start"), reindexers=reindexers
            )|indent(16, false)
            }}
        }
    }
}

void many_thread_mm(
    const {{micro_gemm.get_common_options()['input_t']}}* X,
    const {{micro_gemm.get_common_options()['input_t']}}* W,
    {{micro_gemm.get_common_options()['input_t']}}* Y
    {%- if is_dynamic_M %},
    const int64_t {{kernel.size(GemmOut, -2, unwrapped=True)}}
    {% endif %}
)
{
    {{kernel.maybe_codegen_profile()}}
    constexpr int64_t num_threads = {{num_threads}};
    constexpr int64_t N = {{kernel.size(GemmOut, -1)}};
    constexpr int64_t K = {{kernel.size(X, -1)}};
    constexpr int64_t M0 = {{micro_gemm.register_blocking.block_m}};
    constexpr int64_t N0 = {{micro_gemm.register_blocking.block_n}};
    constexpr int64_t K0 = {{micro_gemm.register_blocking.block_k}};
    constexpr int64_t N0_blocks = (N + N0 - 1) / N0;
    constexpr int64_t K0_blocks = (K + K0 - 1) / K0;

    static_assert(N % N0 == 0, "N dimension must be multiple of N0");

    // TODO(jgong5): improve cache blocking with CPU info (Mc, Kc)
    {%- if is_dynamic_M %}
    const int64_t M = {{kernel.size(GemmOut, -2)}};
    const int64_t M0_blocks = (M + M0 - 1) / M0;
    int64_t Mt_blocks, Nt_blocks, Kt_blocks;
    mm_get_thread_blocking(num_threads, M, N, K, M0, N0, K0, Mt_blocks, Nt_blocks, Kt_blocks);
    const int64_t Mc_blocks = Mt_blocks;
    const int64_t Kc_blocks = Kt_blocks;
    {%- else %}
    constexpr int64_t M = {{kernel.size(GemmOut, -2)}};
    constexpr int64_t M0_blocks = (M + M0 - 1) / M0;
    constexpr int64_t Mt_blocks = {{template.thread_blocking().block_m}};
    constexpr int64_t Nt_blocks = {{template.thread_blocking().block_n}};
    constexpr int64_t Kt_blocks = {{template.thread_blocking().block_k}};
    constexpr int64_t Mc_blocks = {{template.cache_blocking().block_m}};
    constexpr int64_t Kc_blocks = {{template.cache_blocking().block_k}};
    {%- endif %}

    // TODO(jgong5): support k-slicing
    {{kernel.assert_function}}(Kt_blocks == K0_blocks, "Do not support k slicing yet.");
    // make sure all partitions are assigned
    {{kernel.assert_function}}(
        Mt_blocks * Nt_blocks * Kt_blocks * {{num_threads}} >= M0_blocks * N0_blocks * K0_blocks,
        "Not all partitions are assigned."
    );

    #pragma omp parallel num_threads({{num_threads}})
    {
        int tid = omp_get_thread_num();
        int64_t m_block_start, m_block_end, n_block_start, n_block_end, k_block_start, k_block_end;
        mm_get_thread_blocks(
            tid, M0_blocks, N0_blocks, K0_blocks, Mt_blocks, Nt_blocks, Kt_blocks,
            m_block_start, m_block_end, n_block_start, n_block_end, k_block_start, k_block_end);
        for (int64_t mc = m_block_start; mc < m_block_end; mc += Mc_blocks) {
            const int64_t m_start = mc * M0;
            const int64_t m_end = std::min((mc + Mc_blocks) * M0, M);
            const int64_t m_size = m_end - m_start;
            {%- if use_local_acc %}
            {{ kernel.define_buffer(acc_buf_name, [1, "m_end - m_start", "N0"]) }}
            {%- endif %}
            for (int64_t nc = n_block_start; nc < n_block_end; ++nc) {
                const int64_t n_start = nc * N0;
                const int64_t n_size = N0;
                {%- if use_local_acc %}
                {%- set acc = kernel.local_buffers[acc_buf_name] %}
                {%- else %}
                {%- set acc = kernel.slice_nd(GemmOut, [(0, 1), ("m_start", "m_end"), ("n_start", "n_start + N0")]) %}
                {%- endif %}
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * K0;
                    int64_t k_end = std::min((kc + Kc_blocks) * K0, K);
                    {%- set tile_X = kernel.slice_nd(X, [(0, 1), ("m_start", "m_end"), ("k_start", "k_end")]) %}
                    {%- set tile_W = kernel.slice_nd(W, [(0, 1), ("k_start", "k_end"), ("n_start", "n_start + n_size")]) %}
                    if (kc == k_block_start) {
                        {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc, accum=False, is_bmm=is_bmm)|indent(24, false) }}
                    } else {
                        {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc, accum=True, is_bmm=is_bmm)|indent(24, false) }}
                    }
                }
                {%- set tile_Y = kernel.slice_nd(Y_2d, [(0, 1), ("m_start", "m_end"), ("n_start", "n_start + N0")]) %}
                {{ kernel.store_output(
                    tile_Y, acc, GemmOut, epilogue_nodes, offsets=(0, "m_start", "n_start"), reindexers=reindexers
                )|indent(16, false)
                }}
            }
        }
    }
}
"""

MICROKERNEL_DEF = r"""
{{template.header().getvalue()}}

{{micro_gemm.codegen_define(kernel)}}

{{kernel.set_args(inputs={"X": X, "W": W}, outputs={"Y": Y}, aliases=buffer_aliases) }}
"""

SINGLE_THREAD_STUB = r"""
void single_thread_mm(
    const {{micro_gemm.get_common_options()['input_t']}}* X,
    const {{micro_gemm.get_common_options()['input_t']}}* W,
    {{micro_gemm.get_common_options()['output_t']}}* Y
)
"""

BLOCKED_STUB = r"""
void blocked_mm(
    const {{micro_gemm.get_common_options()['input_t']}}* X,
    const {{micro_gemm.get_common_options()['input_t']}}* W,
    {{micro_gemm.get_common_options()['output_t']}}* Y
)
"""

BMM_WRAPPER = r"""
extern "C"
{{kernel.def_kernel(inputs={"X": X, "W": W}, outputs={"Y": Y}, aliases=buffer_aliases)}}
{
    const int64_t B = {{kernel.size(GemmOut, -3, default_value=1, unwrapped=True)}};
    constexpr int64_t num_threads = {{num_threads}};
    int64_t B_single_thread_block = (B / num_threads) * num_threads;
    
    #pragma omp parallel for num_threads({{num_threads}})
    for (int64_t b_start = 0; b_start < B_single_thread_block; ++b_start) {
        one_thread_mm(
            &{{kernel.index(X, ["b_start", 0, 0])}},
            &{{kernel.index(W, ["b_start", 0, 0])}},
            &{{kernel.index(Y, ["b_start", 0, 0])}}
            {%- if is_dynamic_M %},
            {{kernel.size(GemmOut, -2)}}
            {% endif %}
            
        );
    }
    for (int64_t b_start = B_single_thread_block; b_start < B; ++b_start) {
        many_thread_mm(
            &{{kernel.index(X, ["b_start", 0, 0])}},
            &{{kernel.index(W, ["b_start", 0, 0])}},
            &{{kernel.index(Y, ["b_start", 0, 0])}}
            {%- if is_dynamic_M %},
            {{kernel.size(GemmOut, -2)}}
            {% endif %}
        );
    }
}
"""


class CppBmmTemplate(CppTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta=1,
        alpha=1,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
    ):
        assert layout.dtype in [torch.float, torch.bfloat16, torch.half]
        super().__init__(
            "unpacked_bmm", input_nodes, layout, num_threads, epilogue_creator=epilogue_creator
            )
        self.beta = beta
        self.alpha = alpha
        self.num_threads = num_threads
        self.register_blocking = register_blocking
        b, m, n = layout.size
        _, _, k = input_nodes[0].get_size()
        self.b, self.m, self.n, self.k = b, m, n, k
        self.is_dynamic_B = has_free_symbols((b,))
        self.is_dynamic_M = has_free_symbols((m,))

    @cache_on_self
    def thread_blocking(self) -> GemmBlocking:
        # TODO(jgong5): allow tuning various blocking options
        def get_factors(number):
            factors = []
            # priorize more evenly divided factors
            for i in range(int(number**0.5), 0, -1):
                if number % i == 0:
                    factors.append(number // i)
                    factors.append(i)
            return factors

        def get_blocking(num_threads, factor, m_blocks, n_blocks, k_blocks):
            thread_block_n = (n_blocks + factor - 1) // factor
            cofactor = num_threads // factor
            thread_block_m = (m_blocks + cofactor - 1) // cofactor
            return GemmBlocking(thread_block_m, thread_block_n, k_blocks)

        assert (
            not self.is_dynamic_M
        ), "Unable to determine thread blocking for dynamic M."
        register_blocking = self.register_blocking
        m_blocks = (self.m + register_blocking.block_m - 1) // register_blocking.block_m
        n_blocks = (self.n + register_blocking.block_n - 1) // register_blocking.block_n
        k_blocks = (self.k + register_blocking.block_k - 1) // register_blocking.block_k
        factors = get_factors(self.num_threads)
        assert len(factors) > 0
        for factor in factors:
            if n_blocks % factor == 0 and m_blocks % (self.num_threads // factor) == 0:
                return get_blocking(
                    self.num_threads, factor, m_blocks, n_blocks, k_blocks
                )
        for factor in factors:
            if n_blocks % factor == 0:
                return get_blocking(
                    self.num_threads, factor, m_blocks, n_blocks, k_blocks
                )
            cofactor = self.num_threads // factor
            if m_blocks % cofactor == 0:
                return get_blocking(
                    self.num_threads, factor, m_blocks, n_blocks, k_blocks
                )
        raise AssertionError("Should not reach here.")

    @cache_on_self
    def cache_blocking(self) -> GemmBlocking:
        # TODO(jgong5): improve cache blocking with CPU info
        assert (
            not self.is_dynamic_M
        ), "Unable to determine cache blocking for dynamic M."
        thread_blocking = self.thread_blocking()
        return GemmBlocking(thread_blocking.block_m, 1, thread_blocking.block_k)
    
    @staticmethod
    def add_choices(
        choices,
        layout,
        input_nodes,
        beta=1,
        alpha=1,
        trans_w=False,
        input_indices=None,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
    ):
        if input_indices is None:
            input_indices = list(range(len(input_nodes)))

        def reorder_and_filter(inputs, layout_or_out):
            assert len(input_indices) >= 2
            return [inputs[idx] for idx in input_indices], layout_or_out

        def maybe_to_dense(inputs, layout_or_out):
            new_inputs = list(inputs)
            if isinstance(inputs[1], torch.Tensor):
                W = inputs[1]
                new_inputs[1] = W.to_dense() if W.is_mkldnn else W
            return new_inputs, layout_or_out

        def normalize_shapes(inputs, layout_or_out):
            if not trans_w:
                return inputs, layout_or_out

            new_inputs = list(inputs)
            X = inputs[0]
            W = inputs[1]
            B = inputs[2] if len(inputs) > 2 else None
            if isinstance(W, ir.IRNode):
                if trans_w:
                    if not isinstance(W, ir.TensorBox):
                        W = ir.TensorBox(W)
                    W = L.permute(W, [1, 0])
            else:
                if trans_w:
                    assert isinstance(W, torch.Tensor)
                    W = W.transpose(0, 1)
            if B is not None:
                if isinstance(B, ir.IRNode):
                    if not isinstance(B, ir.TensorBox):
                        B = ir.TensorBox(B)
                    B = L.expand(B, (X.get_size()[0], B.get_size()[-1]))
                else:
                    assert isinstance(B, torch.Tensor)
                    B = B.expand(X.shape[0], B.shape[-1])
            new_inputs[1] = W
            if B is not None:
                new_inputs[2] = B
            return new_inputs, layout_or_out

        # TODO(jgong5): decide proper number of threads per problem size
        num_threads = parallel_num_threads()
        new_inputs, _ = normalize_shapes(
            *maybe_to_dense(*reorder_and_filter(input_nodes, layout))
        )
        m, n, k, *_ = mm_args(new_inputs[0], new_inputs[1])
        micro_gemm = create_micro_gemm(
            "micro_gemm",
            m,
            n,
            k,
            input_dtype=layout.dtype,
            output_dtype=torch.float,
            alpha=alpha,
            num_threads=num_threads,
        )
        assert micro_gemm is not None
        _, block_n, _ = micro_gemm.register_blocking

        def pack_weight(inputs, layout_or_out):
            W = inputs[1]
            new_inputs = list(inputs)
            if isinstance(W, ir.IRNode):
                if not isinstance(W, ir.TensorBox):
                    W = ir.TensorBox(W)
                k, n = W.get_size()
                assert (
                    n % block_n == 0
                ), f"The last dimension of W must be a multiple of {block_n}."
                blocked_w = L.permute(
                    L.view(W, (k, n // block_n, block_n)),
                    [1, 0, 2],
                )
                blocked_w = ir.ExternKernel.realize_input(blocked_w)
                blocked_w = ir.ExternKernel.require_contiguous(blocked_w)
                if isinstance(blocked_w, ir.ReinterpretView):
                    # normalize stride to be "contiguous_strides" per size
                    # this avoids the problems in L.view during template codegen
                    assert isinstance(blocked_w.layout, ir.FixedLayout)
                    blocked_w.layout = ir.FixedLayout(
                        blocked_w.layout.device,
                        blocked_w.layout.dtype,
                        blocked_w.layout.size,
                        ir.FlexibleLayout.contiguous_strides(blocked_w.layout.size),
                        blocked_w.layout.offset,
                    )
            else:
                k, n = list(W.shape)
                blocked_w = (
                    W.reshape(k, n // block_n, block_n).transpose(0, 1).contiguous()
                )
                # normalize stride to be "contiguous_strides" per size
                # this avoids the problems in L.view during template codegen
                new_stride = [1]
                for sz in reversed(blocked_w.shape[1:]):
                    new_stride.insert(0, new_stride[0] * sz)
                blocked_w = blocked_w.as_strided(blocked_w.shape, new_stride)
            new_inputs[1] = blocked_w
            return new_inputs, layout_or_out

        def preprocessor(inputs, layout):
            return inputs, layout

        def postprocessor(output):
            return output

        template = DataProcessorTemplateWrapper(
            CppBmmTemplate,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            num_threads=num_threads,
            register_blocking=micro_gemm.register_blocking,
            beta=beta,
            alpha=alpha,
            epilogue_creator=epilogue_creator,
        )
        template.maybe_append_choice(choices)
        return template

    def render(  # type: ignore[override]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> str:
        assert len(self.input_nodes) >= 2

        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node

        if template_buffer_node is not None:
            # Use the updated prepacked weight buffer
            W = template_buffer_node.inputs[1]
            Y = template_buffer_node

        template_buffer = Y
        gemm_output_buffer = template_buffer

        epilogues: List[ir.IRNode] = []
        reindexers: List[Optional[Callable[[List[Any]], List[Any]]]] = []
        if self.epilogue_creator is not None:
            gemm_output_name = "GemmOut"
            gemm_output_buffer = ir.Buffer(gemm_output_name, template_buffer.layout)
            epilogues.append(
                ir.ComputedBuffer(
                    name=template_buffer.get_name(),
                    layout=template_buffer.layout,
                    data=self.epilogue_creator(gemm_output_buffer),
                )
            )
            reindexers.append(None)

        Y_2d: Union[ir.Buffer, ir.ReinterpretView] = Y
        use_local_acc = self.layout.dtype != torch.float
        acc_buf_name = "local_acc_buf"
        if epilogue_nodes:
            epilogues.extend(epilogue_nodes)
            assert Y.get_numel() == epilogues[-1].get_numel()
            Y = cast(ir.Buffer, epilogues[-1])
            if (
                Y.get_size() == template_buffer.get_size()
                and Y.get_stride() == template_buffer.get_stride()
            ):
                reindexers.extend([None] * len(epilogue_nodes))
                Y_2d = Y
            else:
                stride_reversed_order = list(
                    reversed(ir.get_stride_order(Y.get_stride()))
                )
                stride_reindex = ir.same_reorder(stride_reversed_order)
                ordered_size = [Y.get_size()[i] for i in stride_reversed_order]
                reshape_reindex = ir.View.dynamic_reshape_indexer(
                    ordered_size, template_buffer.get_size()
                )
                reindexer = ir.fuse_reindexing(stride_reindex, reshape_reindex)
                reindexers.extend([reindexer] * len(epilogue_nodes))
                if isinstance(Y, ir.BaseView):
                    storage = ir.StorageBox(Y.unwrap_view())
                else:
                    assert isinstance(Y, ir.Buffer)
                    storage = ir.StorageBox(Y)
                Y_2d = ir.ReinterpretView(storage, template_buffer.get_layout())

        micro_gemm = create_micro_gemm(
            f"{kernel.kernel_name}_micro_gemm",
            self.m,
            self.n,
            self.k,
            input_dtype=self.layout.dtype,
            output_dtype=torch.float,
            alpha=self.alpha,
            num_threads=self.num_threads,
        )
        assert micro_gemm is not None
        assert self.register_blocking == micro_gemm.register_blocking

        options = dict(
            X=X,
            W=W,
            inp=None,
            Y=Y,
            GemmOut=gemm_output_buffer,
            buffer_aliases=[(gemm_output_buffer, Y)]
            if gemm_output_buffer is not Y
            else None,
            beta=self.beta,
            alpha=self.alpha,
            num_threads=self.num_threads,
            micro_gemm=micro_gemm,
            is_dynamic_M=self.is_dynamic_M,
            is_packed_weight=False,
            template=self,
            kernel=kernel,
            epilogue_nodes=epilogues,
            reindexers=reindexers,
            Y_2d=Y_2d,
            use_local_acc=use_local_acc,
            acc_buf_name=acc_buf_name,
            is_bmm=True,
        )
        result = self._template_from_string(MICROKERNEL_DEF).render(**options)
        result += self._template_from_string(MM_TEMPLATES).render(**options)
        result += self._template_from_string(BMM_WRAPPER).render(**options)
        return result
