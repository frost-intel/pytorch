# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms FullgraphCompileTest.py onto xccl2.
#
# The original compiles functions containing TorchComm collectives with
# fullgraph=True and asserts both that no graph break occurs and that the
# results are still correct. Those collectives are torchcomms custom ops
# (torch.ops.torchcomms.*), which xccl2 does not register.
#
# The equivalent for a c10d backend is the functional-collective path: inside a
# compiled region dynamo lowers dist.* onto torch.ops._c10d_functional.*, which
# dispatches to whichever backend the group holds. So the same property is under
# test -- collectives compile into one graph and produce the same answer -- via
# the in-tree op set rather than torchcomms'.
#
# The original's 23 methods collapse where c10d has no counterpart:
#   - window_put_get: no Window in torchcomms XCCL either, so not a gap
#   - with_hints_timeout: c10d has no per-call hints
#   - all_gather_v / premul_sum: xccl2 parity gaps, marked below
#   - reduce_scatter_v: c10d folds this into reduce_scatter, covered here

import itertools

import torch
import torch.distributed as dist

from common import (
    dynamo_gap,
    filter_int8_overflow_cases,
    get_op_name,
    is_full_sweep,
    parity_gap,
    run_tests,
    Xccl2TestBase,
)


class FullgraphCompileTest(Xccl2TestBase):
    counts = [4, 1024] if is_full_sweep() else [4]
    dtypes = [torch.float, torch.int] if is_full_sweep() else [torch.float]
    ops = (
        [dist.ReduceOp.SUM, dist.ReduceOp.MAX]
        if is_full_sweep()
        else [dist.ReduceOp.SUM]
    )

    def setUp(self):
        torch._dynamo.reset()
        self.graph_count = 0

    def _backend(self):
        """An inductor-free backend that counts the graphs dynamo produces.

        fullgraph=True already raises on a break; counting is what proves the
        collective was captured rather than run eagerly outside the graph.
        """

        def compiler(gm, example_inputs):
            self.graph_count += 1
            return gm.forward

        return compiler

    def _compile(self, fn):
        return torch.compile(fn, fullgraph=True, backend=self._backend())

    def _input(self, count, dtype):
        return torch.ones(count, dtype=dtype, device=self.device) * (self.rank + 1)

    def _assert_single_graph(self):
        self.assertEqual(
            self.graph_count, 1, "collective did not compile into one graph"
        )

    def _reduce_cases(self):
        cases = list(itertools.product(self.counts, self.dtypes, self.ops))
        return filter_int8_overflow_cases(cases, self.num_ranks, 15)

    def _expected_reduce(self, op):
        n = self.num_ranks
        if op == dist.ReduceOp.SUM:
            return n * (n + 1) // 2
        if op == dist.ReduceOp.MAX:
            return n
        raise RuntimeError(f"Unsupported op: {op}")

    def test_fullgraph_compile_all_reduce(self):
        for count, dtype, op in self._reduce_cases():
            with self.subTest(count=count, dtype=dtype, op=get_op_name(op)):
                torch._dynamo.reset()
                self.graph_count = 0

                def fn(t):
                    dist.all_reduce(t, op=op)
                    return t * 10

                result = self._compile(fn)(self._input(count, dtype))
                self._assert_single_graph()
                expected = self._expected_reduce(op) * 10
                torch.testing.assert_close(
                    result.cpu(), torch.full_like(result.cpu(), expected)
                )

    def test_fullgraph_compile_all_reduce_multiple_calls(self):
        """Several collectives in one graph must not force a break."""
        for count, dtype, op in self._reduce_cases():
            with self.subTest(count=count, dtype=dtype, op=get_op_name(op)):
                torch._dynamo.reset()
                self.graph_count = 0

                def fn(t):
                    dist.all_reduce(t, op=op)
                    dist.all_reduce(t, op=op)
                    return t

                result = self._compile(fn)(self._input(count, dtype))
                self._assert_single_graph()
                if op == dist.ReduceOp.SUM:
                    n = self.num_ranks
                    expected = (n * (n + 1) // 2) * n
                else:
                    expected = self._expected_reduce(op)
                torch.testing.assert_close(
                    result.cpu(), torch.full_like(result.cpu(), expected)
                )

    @dynamo_gap(
        "reduce",
        "dist.reduce builds a pybind11 ReduceOptions, which dynamo cannot "
        "trace (gb0156); fails on every c10d backend, not just xccl2",
    )
    def test_fullgraph_compile_reduce(self):
        for count, dtype, op in self._reduce_cases():
            with self.subTest(count=count, dtype=dtype, op=get_op_name(op)):
                torch._dynamo.reset()
                self.graph_count = 0

                def fn(t):
                    dist.reduce(t, dst=0, op=op)
                    return t

                result = self._compile(fn)(self._input(count, dtype))
                self._assert_single_graph()
                if self.rank == 0:
                    torch.testing.assert_close(
                        result.cpu(),
                        torch.full_like(result.cpu(), self._expected_reduce(op)),
                    )

    @dynamo_gap(
        "broadcast",
        "dist.broadcast builds a pybind11 BroadcastOptions, which dynamo "
        "cannot trace (gb0156); fails on every c10d backend, not just xccl2",
    )
    def test_fullgraph_compile_broadcast(self):
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                torch._dynamo.reset()
                self.graph_count = 0

                def fn(t):
                    dist.broadcast(t, src=0)
                    return t

                result = self._compile(fn)(self._input(count, dtype))
                self._assert_single_graph()
                torch.testing.assert_close(
                    result.cpu(), torch.full_like(result.cpu(), 1)
                )

    def test_fullgraph_compile_all_gather(self):
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                torch._dynamo.reset()
                self.graph_count = 0
                out = torch.empty(
                    count * self.num_ranks, dtype=dtype, device=self.device
                )

                def fn(o, t):
                    dist.all_gather_into_tensor(o, t)
                    return o

                result = self._compile(fn)(out, self._input(count, dtype))
                self._assert_single_graph()
                for r in range(self.num_ranks):
                    chunk = result[r * count : (r + 1) * count]
                    torch.testing.assert_close(
                        chunk.cpu(), torch.full_like(chunk.cpu(), r + 1)
                    )

    def test_fullgraph_compile_reduce_scatter(self):
        for count, dtype, op in self._reduce_cases():
            with self.subTest(count=count, dtype=dtype, op=get_op_name(op)):
                torch._dynamo.reset()
                self.graph_count = 0
                out = torch.empty(count, dtype=dtype, device=self.device)
                inp = torch.ones(
                    count * self.num_ranks, dtype=dtype, device=self.device
                ) * (self.rank + 1)

                def fn(o, t):
                    dist.reduce_scatter_tensor(o, t, op=op)
                    return o

                result = self._compile(fn)(out, inp)
                self._assert_single_graph()
                torch.testing.assert_close(
                    result.cpu(),
                    torch.full_like(result.cpu(), self._expected_reduce(op)),
                )

    def test_fullgraph_compile_all_to_all_single(self):
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                torch._dynamo.reset()
                self.graph_count = 0
                n = self.num_ranks
                out = torch.empty(count * n, dtype=dtype, device=self.device)
                inp = torch.ones(count * n, dtype=dtype, device=self.device) * (
                    self.rank + 1
                )

                def fn(o, t):
                    dist.all_to_all_single(o, t)
                    return o

                result = self._compile(fn)(out, inp)
                self._assert_single_graph()
                for r in range(n):
                    chunk = result[r * count : (r + 1) * count]
                    torch.testing.assert_close(
                        chunk.cpu(), torch.full_like(chunk.cpu(), r + 1)
                    )

    def test_fullgraph_compile_collective_then_compute(self):
        """The collective must fuse with surrounding compute in one graph."""
        torch._dynamo.reset()
        self.graph_count = 0

        def fn(t):
            t = t * 2
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t + 1

        result = self._compile(fn)(self._input(1024, torch.float))
        self._assert_single_graph()
        expected = self.num_ranks * (self.num_ranks + 1) + 1
        torch.testing.assert_close(
            result.cpu(), torch.full_like(result.cpu(), expected)
        )

    @parity_gap(
        "premul_sum",
        "oneCCL lowers PREMUL_SUM to SUM (oneCCL#195/#196); premul_sum_dtypes "
        "is empty for xccl2",
    )
    def test_fullgraph_compile_all_reduce_premul_sum(self):
        def fn(t):
            dist.all_reduce(t, op=dist._make_nccl_premul_sum(2.0))
            return t

        result = self._compile(fn)(self._input(1024, torch.float))
        self._assert_single_graph()
        n = self.num_ranks
        torch.testing.assert_close(
            result.cpu(), torch.full_like(result.cpu(), n * (n + 1))
        )

    @parity_gap(
        "all_gather_v",
        "ProcessGroupXCCL::all_gather rejects mismatched per-rank sizes",
    )
    def test_fullgraph_compile_all_gather_v(self):
        counts = [self.rank + 1] * self.num_ranks
        out = [
            torch.empty(c, dtype=torch.float, device=self.device) for c in counts
        ]

        def fn(o, t):
            dist.all_gather(o, t)
            return o

        self._compile(fn)(out, self._input(self.rank + 1, torch.float))
        self._assert_single_graph()


if __name__ == "__main__":
    run_tests()
