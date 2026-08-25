# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms GatherTest.py onto the xccl2 c10d backend.
# torchcomms' gather(output_list, input, root, async_op) maps to dist.gather,
# which takes the destination list only on the root. The original pins root 0.

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


class GatherTest(CollectiveVariantsMixin, Xccl2TestBase):
    counts = [0, 4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]

    def get_test_cases(self):
        return list(itertools.product(self.counts, self.dtypes))

    def make_operands(self, case):
        count, dtype = case
        input_tensor = torch.ones(count, dtype=dtype, device=self.device) * (
            self.rank + 1
        )
        gather_list = (
            [
                torch.zeros(count, dtype=dtype, device=self.device)
                for _ in range(self.num_ranks)
            ]
            if self.rank == ROOT_RANK
            else None
        )
        return gather_list, input_tensor

    def call(self, operands, case, async_op):
        gather_list, input_tensor = operands
        return dist.gather(
            input_tensor, gather_list, dst=ROOT_RANK, async_op=async_op
        )

    def verify(self, operands, case):
        gather_list, _ = operands
        if self.rank != ROOT_RANK:
            return
        for r, tensor in enumerate(gather_list):
            expected = torch.ones(tensor.numel(), dtype=tensor.dtype) * (r + 1)
            torch.testing.assert_close(
                tensor.cpu(), expected, msg=f"gather output for rank {r}"
            )


if __name__ == "__main__":
    run_tests()
