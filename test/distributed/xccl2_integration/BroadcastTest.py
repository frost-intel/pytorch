# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms BroadcastTest.py onto the xccl2 c10d backend.
# torchcomms' broadcast(tensor, root, async_op) maps to dist.broadcast(src=root).
# The original pins root_rank to 0 in every variant.

import itertools

import torch
import torch.distributed as dist

from common import (
    CollectiveVariantsMixin,
    is_full_sweep,
    run_tests,
    Xccl2TestBase,
)

ROOT_RANK = 0
ROOT_VALUE = 42


class BroadcastTest(CollectiveVariantsMixin, Xccl2TestBase):
    counts = [0, 4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]

    def get_test_cases(self):
        return list(itertools.product(self.counts, self.dtypes))

    def make_operands(self, case):
        count, dtype = case
        fill = ROOT_VALUE if self.rank == ROOT_RANK else 0
        return torch.ones(count, dtype=dtype, device=self.device) * fill

    def call(self, operands, case, async_op):
        return dist.broadcast(operands, src=ROOT_RANK, async_op=async_op)

    def verify(self, operands, case):
        torch.testing.assert_close(
            operands.cpu(),
            torch.full_like(operands.cpu(), ROOT_VALUE),
            msg=f"broadcast from root {ROOT_RANK}",
        )


if __name__ == "__main__":
    run_tests()
