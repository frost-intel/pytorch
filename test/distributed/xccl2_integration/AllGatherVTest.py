# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms AllGatherVTest.py onto the xccl2 c10d backend.
#
# torchcomms exposes all_gather_v for uneven gathers. c10d has a single
# all_gather entry point, so the uneven case is an all_gather whose output list
# has differing sizes; ProcessGroupXCCL::all_gather issues that as one grouped
# broadcast per rank.

import itertools

import torch
import torch.distributed as dist

from common import is_full_sweep, run_tests, Xccl2TestBase


class AllGatherVTest(Xccl2TestBase):
    counts = [4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]

    def get_test_cases(self):
        return list(itertools.product(self.counts, self.dtypes))

    def _operands(self, count, dtype):
        # Rank r contributes (r+1)*count elements.
        sizes = [(r + 1) * count for r in range(self.num_ranks)]
        input_tensor = torch.ones(
            sizes[self.rank], dtype=dtype, device=self.device
        ) * (self.rank + 1)
        output_list = [
            torch.zeros(sizes[r], dtype=dtype, device=self.device)
            for r in range(self.num_ranks)
        ]
        return output_list, input_tensor

    def _verify(self, output_list):
        for r, tensor in enumerate(output_list):
            torch.testing.assert_close(
                tensor.cpu(),
                torch.full_like(tensor.cpu(), r + 1),
                msg=f"uneven all_gather output for rank {r}",
            )

    def test_sync_all_gather_v(self):
        for count, dtype in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype):
                output_list, input_tensor = self._operands(count, dtype)
                dist.all_gather(output_list, input_tensor, async_op=False)
                self._verify(output_list)

    def test_async_all_gather_v(self):
        for count, dtype in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype):
                output_list, input_tensor = self._operands(count, dtype)
                work = dist.all_gather(output_list, input_tensor, async_op=True)
                work.wait()
                self._verify(output_list)


if __name__ == "__main__":
    run_tests()
