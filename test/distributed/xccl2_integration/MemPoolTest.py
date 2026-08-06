# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms MemPoolTest.py onto xccl2.
#
# The original allocates tensors inside a MemPool backed by the backend's own
# allocator, then reduces them, checking that comm-registered memory works as
# normal tensor storage. torchcomms XCCL supplies that allocator by registering
# an allocator factory over onecclMemAlloc (TorchCommXCCL.cpp).
#
# xccl2 exposes the same oneCCL memory through c10d's own hook instead:
# ProcessGroupXCCL::getMemAllocator(), reached from Python as the
# Backend.mem_allocator property, exactly as nccl2 does.

import torch
import torch.distributed as dist

from common import run_tests, Xccl2TestBase

TENSOR_SIZE = 1024 * 1024
NUM_GROUPS = 16


class MemPoolTest(Xccl2TestBase):
    def _input_tensor(self):
        return torch.ones(TENSOR_SIZE, device=self.device) * float(self.rank + 1)

    def _verify(self, tensor):
        expected = self.num_ranks * (self.num_ranks + 1) // 2
        torch.testing.assert_close(
            tensor.cpu(), torch.full_like(tensor.cpu(), float(expected))
        )

    def test_mem_pool(self) -> None:
        device_module = torch.get_device_module(self.device)
        backend = dist.group.WORLD._get_backend(self.device)
        allocator = backend.mem_allocator

        groups = [
            dist.new_group(ranks=list(range(self.num_ranks)))
            for _ in range(NUM_GROUPS)
        ]
        try:
            tensors = []
            for group in groups:
                pool = device_module.MemPool(allocator)
                with device_module.use_mem_pool(pool):
                    tensor = self._input_tensor()
                tensors.append(tensor)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

            device_module.current_stream().synchronize()
            for tensor in tensors:
                self._verify(tensor)
        finally:
            for group in groups:
                dist.destroy_process_group(group)


if __name__ == "__main__":
    run_tests()
