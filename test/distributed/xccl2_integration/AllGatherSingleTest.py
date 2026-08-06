# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms AllGatherSingleTest.py onto the xccl2 c10d backend.
# torchcomms' all_gather_single maps to dist.all_gather_into_tensor; xccl2
# implements it as ProcessGroupXCCL::allGatherSingleImpl.

import itertools

import torch
import torch.distributed as dist

from common import (
    CollectiveVariantsMixin,
    is_full_sweep,
    run_tests,
    Xccl2TestBase,
)


class AllGatherSingleTest(CollectiveVariantsMixin, Xccl2TestBase):
    counts = [0, 4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]

    def get_test_cases(self):
        return list(itertools.product(self.counts, self.dtypes))

    def make_operands(self, case):
        count, dtype = case
        input_tensor = torch.ones(count, dtype=dtype, device=self.device) * (
            self.rank + 1
        )
        output_tensor = torch.zeros(
            count * self.num_ranks, dtype=dtype, device=self.device
        )
        return output_tensor, input_tensor

    def call(self, operands, case, async_op):
        output_tensor, input_tensor = operands
        return dist.all_gather_into_tensor(
            output_tensor, input_tensor, async_op=async_op
        )

    def verify(self, operands, case):
        count, dtype = case
        output_tensor, _ = operands
        if count == 0:
            return
        output_cpu = output_tensor.cpu()
        for r in range(self.num_ranks):
            section = output_cpu[r * count : (r + 1) * count]
            expected = torch.ones(count, dtype=dtype) * (r + 1)
            torch.testing.assert_close(
                section, expected, msg=f"all_gather_single section for rank {r}"
            )


if __name__ == "__main__":
    run_tests()
