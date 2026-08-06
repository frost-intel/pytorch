# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms ReduceScatterTest.py onto the xccl2 c10d backend.
# torchcomms' reduce_scatter(output, input_list, op, async_op) maps to
# dist.reduce_scatter. Every rank builds the same input list (entry r holds
# r+1), so rank r's output is the reduction of r+1 across all ranks.

import itertools

import torch
import torch.distributed as dist

from common import (
    CollectiveVariantsMixin,
    filter_int8_overflow_cases,
    get_op_name,
    is_full_sweep,
    run_tests,
    Xccl2TestBase,
)


class ReduceScatterTest(CollectiveVariantsMixin, Xccl2TestBase):
    counts = [0, 4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]
    ops = (
        [dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.AVG]
        if is_full_sweep()
        else [dist.ReduceOp.SUM]
    )

    def get_test_cases(self):
        cases = list(itertools.product(self.counts, self.dtypes, self.ops))
        return filter_int8_overflow_cases(cases, self.num_ranks, 15)

    def make_operands(self, case):
        count, dtype, _ = case
        input_tensors = [
            torch.ones(count, dtype=dtype, device=self.device) * (r + 1)
            for r in range(self.num_ranks)
        ]
        output_tensor = torch.zeros(count, dtype=dtype, device=self.device)
        return output_tensor, input_tensors

    def call(self, operands, case, async_op):
        output_tensor, input_tensors = operands
        _, _, op = case
        return dist.reduce_scatter(
            output_tensor, input_tensors, op=op, async_op=async_op
        )

    def verify(self, operands, case):
        output_tensor, _ = operands
        _, _, op = case
        if op == dist.ReduceOp.SUM:
            expected = self.num_ranks * (self.rank + 1)
        elif op in (dist.ReduceOp.MAX, dist.ReduceOp.AVG):
            expected = self.rank + 1
        else:
            raise RuntimeError(f"Unsupported reduce operation: {op}")

        torch.testing.assert_close(
            output_tensor.cpu(),
            torch.full_like(output_tensor.cpu(), expected),
            msg=f"reduce_scatter with op {get_op_name(op)}",
        )


if __name__ == "__main__":
    run_tests()
