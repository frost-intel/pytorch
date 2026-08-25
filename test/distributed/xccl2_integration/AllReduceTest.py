# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms AllReduceTest.py onto the xccl2 c10d backend.
#
# The original drives torchcomms' TorchComm.all_reduce(tensor, op, async_op),
# which returns a work handle for both sync and async calls. c10d returns a
# handle only when async_op=True, so the "sync with work object" and "sync
# without work object" variants of the original collapse; both are kept, with
# the sync variant asserting that no handle comes back.
#
# The original's CUDA-graph variants are ncclx-only and have no xccl2 analogue.

import itertools

import torch
import torch.distributed as dist

from common import (
    CollectiveVariantsMixin,
    filter_int8_overflow_cases,
    get_op_name,
    is_full_sweep,
    parity_gap,
    run_tests,
    Xccl2TestBase,
)


class AllReduceTest(CollectiveVariantsMixin, Xccl2TestBase):
    counts = [0, 4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]
    ops = (
        [dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.AVG]
        if is_full_sweep()
        else [dist.ReduceOp.SUM]
    )

    def get_test_cases(self):
        cases = list(itertools.product(self.counts, self.dtypes, self.ops))
        # sum(1..16) = 136 overflows int8, so cap the rank count for int8 SUM/AVG.
        return filter_int8_overflow_cases(cases, self.num_ranks, 15)

    def make_operands(self, case):
        count, dtype, _ = case
        return torch.ones(count, dtype=dtype, device=self.device) * (self.rank + 1)

    def call(self, operands, case, async_op):
        _, _, op = case
        return dist.all_reduce(operands, op=op, async_op=async_op)

    def verify(self, operands, case):
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
            msg=lambda error: f"all_reduce with op {get_op_name(op)}: {error}",
        )

    @parity_gap(
        "premul_sum",
        "oneCCL lowers PREMUL_SUM to SUM (oneCCL#195/#196); "
        "premul_sum_dtypes is empty for xccl2 in c10d_backend_common.py",
    )
    def test_premul_sum_all_reduce(self):
        """PREMUL_SUM sweep from the original suite."""
        for count, dtype in itertools.product(
            self.counts, [torch.half, torch.float, torch.double, torch.bfloat16]
        ):
            with self.subTest(count=count, dtype=dtype):
                op = dist._make_nccl_premul_sum(2.0)
                tensor = self.make_operands((count, dtype, op))
                dist.all_reduce(tensor, op=op, async_op=False)
                n = self.num_ranks
                torch.testing.assert_close(
                    tensor.cpu(), torch.full_like(tensor.cpu(), n * (n + 1))
                )


if __name__ == "__main__":
    run_tests()
