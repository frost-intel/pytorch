# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms ScatterTest.py onto the xccl2 c10d backend.
# torchcomms' scatter(output, input_list, root, async_op) maps to dist.scatter,
# which takes the source list only on the root. The original pins root 0.

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


class ScatterTest(CollectiveVariantsMixin, Xccl2TestBase):
    counts = [0, 4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]

    def get_test_cases(self):
        return list(itertools.product(self.counts, self.dtypes))

    def make_operands(self, case):
        count, dtype = case
        # Entry r of the root's list is destined for rank r and holds r+1.
        scatter_list = (
            [
                torch.ones(count, dtype=dtype, device=self.device) * (r + 1)
                for r in range(self.num_ranks)
            ]
            if self.rank == ROOT_RANK
            else None
        )
        output_tensor = torch.zeros(count, dtype=dtype, device=self.device)
        return output_tensor, scatter_list

    def call(self, operands, case, async_op):
        output_tensor, scatter_list = operands
        return dist.scatter(
            output_tensor, scatter_list, src=ROOT_RANK, async_op=async_op
        )

    def verify(self, operands, case):
        output_tensor, _ = operands
        torch.testing.assert_close(
            output_tensor.cpu(),
            torch.full_like(output_tensor.cpu(), self.rank + 1),
            msg=f"scatter output on rank {self.rank}",
        )


if __name__ == "__main__":
    run_tests()
