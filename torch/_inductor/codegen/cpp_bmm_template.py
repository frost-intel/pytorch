# mypy: allow-untyped-defs
from typing import Any, Callable, cast, List, Optional, Union

import torch
import torch.utils
from .. import ir, lowering as L

from ..kernel.mm_common import mm_args
from ..select_algorithm import DataProcessorTemplateWrapper
from ..mkldnn_lowerings import create_epilogue_with_attr
from ..utils import parallel_num_threads
from ..virtualized import V
from .cpp_micro_gemm import create_micro_gemm
from .cpp_template import CppTemplate
from .cpp_template_kernel import parse_expr_with_index_symbols

from .cpp_template_kernel import CppTemplateKernel
from .cpp_gemm_template import GEMM_TEMPLATE, CppPackedGemmTemplate
from .cpp_utils import DTYPE_TO_CPP, GemmBlocking, LocalBufferScope

MICROKERNEL_DEF = r"""
{{template.header().getvalue()}}

{{micro_gemm.codegen_define(kernel)}}
"""

SINGLE_THREAD_STUB = r"""
void single_thread_mm(
    const {{micro_gemm.get_common_options()['input_t']}}* X,
    const {{micro_gemm.get_common_options()['input_t']}}* W,
    {{micro_gemm.get_common_options()['input_t']}}* Y
    {%- if is_dynamic_M %},
    const int64_t {{kernel.size(GemmOut, -2, unwrapped=True)}}
    {% endif %}
)
"""

BLOCKED_STUB = r"""
void blocked_mm(
    const {{micro_gemm.get_common_options()['input_t']}}* X,
    const {{micro_gemm.get_common_options()['input_t']}}* W,
    {{micro_gemm.get_common_options()['input_t']}}* Y
    {%- if is_dynamic_M %},
    const int64_t {{kernel.size(GemmOut, -2, unwrapped=True)}}
    {% endif %}
)
"""

BMM_WRAPPER = r"""
extern "C"
{{kernel.def_kernel(inputs={"BX": BX, "BW": BW}, outputs={"BY": BY}, aliases=buffer_aliases)}}
{
    const int64_t B = {{kernel.size(BY, -3, unwrapped=True)}};
    {%- if num_threads > 1 %}
    constexpr int64_t num_threads = {{num_threads}};
    int64_t B_single_thread_block = (B / num_threads) * num_threads;
    
    #pragma omp parallel for num_threads({{num_threads}})
    {%- else %}
    int64_t B_single_thread_block = B;
    {%- endif %}
    for (int64_t b_start = 0; b_start < B_single_thread_block; ++b_start) {
        single_thread_mm(
            &{{kernel.index(BX, ["b_start", 0, 0])}},
            &{{kernel.index(BW, ["b_start", 0, 0])}},
            &{{kernel.index(BY, ["b_start", 0, 0])}}
            {%- if is_dynamic_M %},
            {{kernel.size(GemmOut, -2)}}
            {% endif %}
        );
    }
    for (int64_t b_start = B_single_thread_block; b_start < B; ++b_start) {
        blocked_mm(
            &{{kernel.index(BX, ["b_start", 0, 0])}},
            &{{kernel.index(BW, ["b_start", 0, 0])}},
            &{{kernel.index(BY, ["b_start", 0, 0])}}
            {%- if is_dynamic_M %},
            {{kernel.size(GemmOut, -2)}}
            {% endif %}
        );
    }
}
"""


class CppBmmTemplate(CppPackedGemmTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta=1,
        alpha=1,
        has_bias=False,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
        name="bmm"
    ):
        super().__init__(
            input_nodes,
            layout,
            num_threads,
            register_blocking,
            beta=beta,
            alpha=alpha,
            has_bias=has_bias,
            epilogue_creator=epilogue_creator,
            name=name
        )
        self.should_pack_weights = False

    @staticmethod
    def add_choices(
        choices,
        layout,
        input_nodes,
        beta=1,
        alpha=1,
        has_bias=False,
        trans_w=False,
        input_indices=None,
        should_pack_weights=False,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
    ):
        options = super(CppBmmTemplate, CppBmmTemplate)._get_params_for_choices(
            layout=layout,
            input_nodes=input_nodes,
            beta=beta,
            alpha=alpha,
            has_bias=has_bias,
            trans_w=trans_w,
            input_indices=input_indices,
            should_pack_weights=should_pack_weights,
            epilogue_creator=epilogue_creator
        )
        template = DataProcessorTemplateWrapper(
            CppBmmTemplate,
            **options
        )
        template.maybe_append_choice(choices)
        return template

    def get_options(self, kernel, template_buffer_node, epilogue_nodes, **kwargs):
        options = super().get_options(kernel, template_buffer_node, epilogue_nodes, **kwargs)
        options['should_pack_weights'] = self.should_pack_weights
        BX, BW, BY = options['X'], options['W'], options['Y']
        options['BX'], options['BW'], options['BY'] = BX, BW, BY
        for kword in ['X', 'W', 'Y', 'GemmOut', 'Y_2d']:
            if isinstance(options[kword], list):
                for i in range(len(options[kword])):
                    node = options['Y']
                    #pointwise = create_epilogue_with_attr(node, 'relu')
                    #pointwise.origin_node = options[kword][i].origin_node
                    #pointwise.origins = options[kword][i].origins
                    #new_node = ir.ComputedBuffer(
                    #    name=options[kword][i].name,
                    #    layout=node.layout,
                    #    data=pointwise
                    #)
                    #options[kword][i] = new_node

                    #kernel.select(options[kword][i], 0, 0)
                #options[kword] = [
                #    cast(ir.ComputedBuffer, kernel.select(options[kword][i], 0, 0)) for i in range(len(options[kword]))
                #]
            else:
                options[kword] = kernel.select(options[kword], 0, 0)
        return options

    def render(  # type: ignore[override]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> str:
        options = self.get_options(kernel, template_buffer_node, epilogue_nodes, **kwargs)
        BX, BW, BY = options['BX'], options['BW'], options['BY']
        X, W, Y = options['X'], options['W'], options['Y']
        buffer_aliases = options['buffer_aliases']

        kernel.set_args(inputs={"X": X, "W": W}, outputs={"Y": Y}, aliases=buffer_aliases)
        result = self._template_from_string(MICROKERNEL_DEF).render(**options)
        result += self._template_from_string(BLOCKED_STUB+GEMM_TEMPLATE).render(**options)
        self.thread_blocking.clear_cache(self)
        self.cache_blocking.clear_cache(self)
        tmp_num_threads = self.num_threads
        self.num_threads = 1
        result += self._template_from_string(SINGLE_THREAD_STUB+GEMM_TEMPLATE).render(**{**options, 'num_threads': 1})
        self.thread_blocking.clear_cache(self)
        self.cache_blocking.clear_cache(self)
        self.num_threads = tmp_num_threads
        kernel.set_args(inputs={"BX": BX, "BW": BW}, outputs={"BY": BY}, aliases=buffer_aliases)
        result += self._template_from_string(BMM_WRAPPER).render(**options)
        return result
