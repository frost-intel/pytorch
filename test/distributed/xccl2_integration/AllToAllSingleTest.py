# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms AllToAllSingleTest.py onto the xccl2 c10d backend.
# torchcomms' all_to_all_single maps to dist.all_to_all_single with no split
# sizes; xccl2 implements it as ProcessGroupXCCL::allToAllSingleImpl.

import itertools

import torch
import torch.distributed as dist

from common import (
    CollectiveVariantsMixin,
    is_full_sweep,
    run_tests,
    Xccl2TestBase,
)


class AllToAllSingleTest(CollectiveVariantsMixin, Xccl2TestBase):
    counts = [0, 4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]

    def get_test_cases(self):
        return list(itertools.product(self.counts, self.dtypes))

    def make_operands(self, case):
        count, dtype = case
        # Section i of the input is destined for rank i.
        input_tensor = torch.zeros(
            count * self.num_ranks, dtype=dtype, device=self.device
        )
        for i in range(self.num_ranks):
            input_tensor[i * count : (i + 1) * count].fill_(self.rank + 1)
        output_tensor = torch.zeros(
            count * self.num_ranks, dtype=dtype, device=self.device
        )
        return output_tensor, input_tensor

    def call(self, operands, case, async_op):
        output_tensor, input_tensor = operands
        return dist.all_to_all_single(
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
                section, expected, msg=f"all_to_all_single section from rank {r}"
            )


if __name__ == "__main__":
    run_tests()
