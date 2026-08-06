# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms ReduceTest.py onto the xccl2 c10d backend.
# torchcomms' reduce(tensor, root, op, async_op) maps to dist.reduce(dst=root).
# The original pins root_rank to 0 and only checks the result on the root; c10d
# leaves the non-root buffers unspecified, so the same restriction applies here.

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

ROOT_RANK = 0


class ReduceTest(CollectiveVariantsMixin, Xccl2TestBase):
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
        return torch.ones(count, dtype=dtype, device=self.device) * (self.rank + 1)

    def call(self, operands, case, async_op):
        _, _, op = case
        return dist.reduce(operands, dst=ROOT_RANK, op=op, async_op=async_op)

    def verify(self, operands, case):
        if self.rank != ROOT_RANK:
            return
        _, _, op = case
        n = self.num_ranks
        if op == dist.ReduceOp.SUM:
            expected = n * (n + 1) // 2
        elif op == dist.ReduceOp.MAX:
            expected = n
        elif op == dist.ReduceOp.AVG:
            expected = (n * (n + 1) / 2) / n
        else:
            raise RuntimeError(f"Unsupported reduce operation: {op}")

        torch.testing.assert_close(
            operands.cpu(),
            torch.full_like(operands.cpu(), expected),
            msg=f"reduce with op {get_op_name(op)}",
        )


if __name__ == "__main__":
    run_tests()
