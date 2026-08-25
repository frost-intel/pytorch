# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms WaitBlockingTest.py onto the xccl2 c10d backend.
#
# torchcomms exposes work.wait_blocking(), which blocks the host until the op
# retires. c10d Work has no such entry point: work.wait() is stream-ordered,
# and blocking the host means following it with a device synchronize. The
# host-blocking cases below are expressed that way; the test that asserts
# wait_blocking's own semantics has no xccl2 analogue.

import torch
import torch.distributed as dist

from common import parity_gap, run_tests, Xccl2TestBase


class WaitBlockingTest(Xccl2TestBase):
    def _input(self, count=1024 * 1024):
        return torch.ones(count, device=self.device) * (self.rank + 1)

    def _expected(self, tensor):
        n = self.num_ranks
        return torch.full_like(tensor.cpu(), n * (n + 1) / 2)

    def test_wait_then_sync_blocks_until_complete(self):
        """wait() followed by a device sync must leave the result readable."""
        tensor = self._input()
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
        self.sync_device()
        torch.testing.assert_close(tensor.cpu(), self._expected(tensor))

    def test_wait_multiple_ops(self):
        """A batch of outstanding ops must all retire correctly."""
        tensors = [self._input(1024) for _ in range(8)]
        works = [
            dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True) for t in tensors
        ]
        for work in works:
            work.wait()
        self.sync_device()
        for tensor in tensors:
            torch.testing.assert_close(tensor.cpu(), self._expected(tensor))

    def test_wait_idempotent(self):
        """Waiting more than once must be harmless."""
        tensor = self._input()
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
        work.wait()
        work.wait()
        self.sync_device()
        torch.testing.assert_close(tensor.cpu(), self._expected(tensor))

    def test_is_completed_after_wait(self):
        """A waited-on handle must report completion."""
        tensor = self._input()
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
        self.sync_device()
        self.assertTrue(work.is_completed())

    @parity_gap(
        "wait_blocking",
        "torchcomms Work.wait_blocking() blocks the host; c10d Work.wait() is "
        "stream-ordered and xccl2 exposes no host-blocking wait",
    )
    def test_wait_blocking_semantics(self):
        tensor = self._input()
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        work.wait_blocking()
        torch.testing.assert_close(tensor.cpu(), self._expected(tensor))


if __name__ == "__main__":
    run_tests()
