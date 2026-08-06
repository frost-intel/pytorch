# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms ReduceScatterVTest.py onto the xccl2 c10d backend.
#
# torchcomms splits even and uneven reduce-scatter into reduce_scatter and
# reduce_scatter_v. c10d has a single reduce_scatter entry point, so the uneven
# case is expressed as a reduce_scatter whose input list has differing sizes.
# oneCCL has no reduce_scatterv, so xccl2 lowers this to one grouped reduce per
# rank; this file is the acceptance test for that path.

import itertools

import torch
import torch.distributed as dist

from common import (
    filter_int8_overflow_cases,
    get_op_name,
    is_full_sweep,
    run_tests,
    Xccl2TestBase,
)


class ReduceScatterVTest(Xccl2TestBase):
    counts = [4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]
    ops = (
        [dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.AVG]
        if is_full_sweep()
        else [dist.ReduceOp.SUM]
    )

    def get_test_cases(self):
        cases = list(itertools.product(self.counts, self.dtypes, self.ops))
        return filter_int8_overflow_cases(cases, self.num_ranks, 15)

    def _sizes(self, count):
        """Per-rank element counts: rank r receives (r+1)*count elements."""
        return [(r + 1) * count for r in range(self.num_ranks)]

    def _operands(self, count, dtype, include_empty=False):
        sizes = self._sizes(count)
        if include_empty:
            # Empty per-rank inputs take the skip path inside the grouped reduce.
            sizes = [0 if r % 2 == 0 else sizes[r] for r in range(self.num_ranks)]
        input_list = [
            torch.ones(sizes[r], dtype=dtype, device=self.device) * (r + 1)
            for r in range(self.num_ranks)
        ]
        output = torch.zeros(sizes[self.rank], dtype=dtype, device=self.device)
        return output, input_list

    def _verify(self, output, op):
        if op == dist.ReduceOp.SUM:
            expected = self.num_ranks * (self.rank + 1)
        elif op in (dist.ReduceOp.MAX, dist.ReduceOp.AVG):
            expected = self.rank + 1
        else:
            raise RuntimeError(f"Unsupported reduce operation: {op}")

        torch.testing.assert_close(
            output.cpu(),
            torch.full_like(output.cpu(), expected),
            msg=f"uneven reduce_scatter with op {get_op_name(op)}",
        )

    def test_sync_reduce_scatter_v(self):
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                output, input_list = self._operands(count, dtype)
                dist.reduce_scatter(output, input_list, op=op, async_op=False)
                self._verify(output, op)

    def test_async_reduce_scatter_v(self):
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                output, input_list = self._operands(count, dtype)
                work = dist.reduce_scatter(output, input_list, op=op, async_op=True)
                work.wait()
                self._verify(output, op)

    def test_reduce_scatter_v_with_empty_inputs(self):
        """Ranks contributing zero elements must be skipped, not hang.

        oneCCL hangs when a non-root rank passes a null receive buffer, so the
        xccl2 grouped-reduce path points non-root ranks at their own input.
        """
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                output, input_list = self._operands(count, dtype, include_empty=True)
                dist.reduce_scatter(output, input_list, op=op, async_op=False)
                if output.numel() > 0:
                    self._verify(output, op)

    def test_reduce_scatter_v_rejects_mismatched_local_size(self):
        """The local rank's input must still match the output size."""
        input_list = [
            torch.ones(4, device=self.device) for _ in range(self.num_ranks)
        ]
        output = torch.zeros(8, device=self.device)
        with self.assertRaises(Exception):
            dist.reduce_scatter(output, input_list, async_op=False)


if __name__ == "__main__":
    run_tests()
