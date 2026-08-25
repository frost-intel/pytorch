# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms BarrierTest.py onto the xccl2 c10d backend.

import torch
import torch.distributed as dist

from common import run_tests, Xccl2TestBase


class BarrierTest(Xccl2TestBase):
    num_replays = 4

    def test_sync_barrier(self):
        """Synchronous barrier returns no handle."""
        work = dist.barrier(async_op=False)
        self.assertIsNone(work)

    def test_sync_barrier_no_work(self):
        """Synchronous barrier with the return value discarded."""
        dist.barrier(async_op=False)

    def test_async_barrier(self):
        """Asynchronous barrier waited to completion."""
        work = dist.barrier(async_op=True)
        work.wait()

    def test_async_barrier_early_reset(self):
        """Dropping the handle after waiting must not hang the next barrier."""
        work = dist.barrier(async_op=True)
        work.wait()
        work = None
        dist.barrier()

    def test_repeated_barriers(self):
        """Back-to-back barriers must not deadlock."""
        for _ in range(self.num_replays):
            dist.barrier()

    def test_barrier_orders_pending_collective(self):
        """A barrier must not retire before prior work on the same PG."""
        tensor = torch.ones(1024, device=self.device) * (self.rank + 1)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        dist.barrier()
        self.sync_device()

        n = self.num_ranks
        torch.testing.assert_close(
            tensor.cpu(), torch.full_like(tensor.cpu(), n * (n + 1) / 2)
        )


if __name__ == "__main__":
    run_tests()
