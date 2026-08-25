# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms ReduceScatterSingleTest.py onto the xccl2 c10d backend.
# torchcomms' reduce_scatter_single maps to dist.reduce_scatter_tensor; xccl2
# implements it as ProcessGroupXCCL::reduceScatterSingleImpl.

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


class ReduceScatterSingleTest(CollectiveVariantsMixin, Xccl2TestBase):
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
        input_tensor = torch.zeros(
            count * self.num_ranks, dtype=dtype, device=self.device
        )
        for r in range(self.num_ranks):
            input_tensor[r * count : (r + 1) * count].fill_(r + 1)
        output_tensor = torch.zeros(count, dtype=dtype, device=self.device)
        return output_tensor, input_tensor

    def call(self, operands, case, async_op):
        output_tensor, input_tensor = operands
        _, _, op = case
        return dist.reduce_scatter_tensor(
            output_tensor, input_tensor, op=op, async_op=async_op
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
            msg=f"reduce_scatter_single with op {get_op_name(op)}",
        )


if __name__ == "__main__":
    run_tests()
