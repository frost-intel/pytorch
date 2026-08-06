# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms AllToAllTest.py onto the xccl2 c10d backend.
# torchcomms' all_to_all(output_list, input_list, async_op) maps to
# dist.all_to_all. Every entry of this rank's input list holds rank+1, so the
# entry received from rank r holds r+1.

import itertools

import torch
import torch.distributed as dist

from common import (
    CollectiveVariantsMixin,
    is_full_sweep,
    run_tests,
    Xccl2TestBase,
)


class AllToAllTest(CollectiveVariantsMixin, Xccl2TestBase):
    counts = [0, 4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]

    def get_test_cases(self):
        return list(itertools.product(self.counts, self.dtypes))

    def make_operands(self, case):
        count, dtype = case
        input_tensors = [
            torch.ones(count, dtype=dtype, device=self.device) * (self.rank + 1)
            for _ in range(self.num_ranks)
        ]
        output_tensors = [
            torch.zeros(count, dtype=dtype, device=self.device)
            for _ in range(self.num_ranks)
        ]
        return output_tensors, input_tensors

    def call(self, operands, case, async_op):
        output_tensors, input_tensors = operands
        return dist.all_to_all(output_tensors, input_tensors, async_op=async_op)

    def verify(self, operands, case):
        output_tensors, _ = operands
        for r, tensor in enumerate(output_tensors):
            expected = torch.ones(tensor.numel(), dtype=tensor.dtype) * (r + 1)
            torch.testing.assert_close(
                tensor.cpu(), expected, msg=f"all_to_all output from rank {r}"
            )


if __name__ == "__main__":
    run_tests()
