# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms AllGatherTest.py onto the xccl2 c10d backend.
# torchcomms' all_gather(output_list, input, async_op) maps to dist.all_gather.

import itertools

import torch
import torch.distributed as dist

from common import (
    CollectiveVariantsMixin,
    is_full_sweep,
    run_tests,
    Xccl2TestBase,
)


class AllGatherTest(CollectiveVariantsMixin, Xccl2TestBase):
    counts = [0, 4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]

    def get_test_cases(self):
        return list(itertools.product(self.counts, self.dtypes))

    def make_operands(self, case):
        count, dtype = case
        input_tensor = torch.ones(count, dtype=dtype, device=self.device) * (
            self.rank + 1
        )
        output_tensors = [
            torch.zeros(count, dtype=dtype, device=self.device)
            for _ in range(self.num_ranks)
        ]
        return output_tensors, input_tensor

    def call(self, operands, case, async_op):
        output_tensors, input_tensor = operands
        return dist.all_gather(output_tensors, input_tensor, async_op=async_op)

    def verify(self, operands, case):
        output_tensors, _ = operands
        for i, tensor in enumerate(output_tensors):
            expected = torch.ones(tensor.numel(), dtype=tensor.dtype) * (i + 1)
            torch.testing.assert_close(
                tensor.cpu(), expected, msg=f"all_gather output for rank {i}"
            )


if __name__ == "__main__":
    run_tests()
