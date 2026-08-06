# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms SendRecvTest.py onto the xccl2 c10d backend.
# torchcomms' send/recv(tensor, peer, async_op) map to dist.send/isend and
# dist.recv/irecv. Each rank sends to rank+1 and receives from rank-1; even
# ranks send before receiving and odd ranks receive first, which is what keeps
# the blocking variants from deadlocking.

import itertools

import torch
import torch.distributed as dist

from common import is_full_sweep, run_tests, Xccl2TestBase


class SendRecvTest(Xccl2TestBase):
    counts = [0, 4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]

    def get_test_cases(self):
        return list(itertools.product(self.counts, self.dtypes))

    @property
    def send_rank(self):
        return (self.rank + 1) % self.num_ranks

    @property
    def recv_rank(self):
        return (self.rank - 1) % self.num_ranks

    def _operands(self, count, dtype):
        send_tensor = torch.ones(count, dtype=dtype, device=self.device) * (
            self.rank + 1
        )
        recv_tensor = torch.zeros(count, dtype=dtype, device=self.device)
        return send_tensor, recv_tensor

    def _verify(self, recv_tensor):
        expected = torch.full_like(recv_tensor.cpu(), self.recv_rank + 1)
        torch.testing.assert_close(
            recv_tensor.cpu(),
            expected,
            msg=f"recv on rank {self.rank} from rank {self.recv_rank}",
        )

    def test_sync_send_recv(self):
        """Blocking send/recv around the ring."""
        for count, dtype in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype):
                send_tensor, recv_tensor = self._operands(count, dtype)
                if self.rank % 2 == 0:
                    dist.send(send_tensor, self.send_rank)
                    dist.recv(recv_tensor, self.recv_rank)
                else:
                    dist.recv(recv_tensor, self.recv_rank)
                    dist.send(send_tensor, self.send_rank)
                self._verify(recv_tensor)

    def _issue_async(self, send_tensor, recv_tensor):
        """Issue ordered by rank parity.

        Ungrouped bidirectional p2p is enqueued back-to-back on one stream, so
        an unordered ring deadlocks on oneCCL (stock xccl does this too). The
        original alternates by parity for the same reason.
        """
        if self.rank % 2 == 0:
            send_work = dist.isend(send_tensor, self.send_rank)
            recv_work = dist.irecv(recv_tensor, self.recv_rank)
        else:
            recv_work = dist.irecv(recv_tensor, self.recv_rank)
            send_work = dist.isend(send_tensor, self.send_rank)
        return send_work, recv_work

    def test_async_send_recv(self):
        """Non-blocking send/recv waited to completion."""
        for count, dtype in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype):
                send_tensor, recv_tensor = self._operands(count, dtype)
                send_work, recv_work = self._issue_async(send_tensor, recv_tensor)
                send_work.wait()
                recv_work.wait()
                self._verify(recv_tensor)

    def test_async_send_recv_early_reset(self):
        """Dropping the handles after waiting must not affect the result."""
        for count, dtype in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype):
                send_tensor, recv_tensor = self._operands(count, dtype)
                send_work, recv_work = self._issue_async(send_tensor, recv_tensor)
                send_work.wait()
                recv_work.wait()
                send_work = None
                recv_work = None
                self._verify(recv_tensor)

    def test_send_recv_input_deleted(self):
        """Dropping the buffers right after enqueue must not crash.

        xccl2 sets supports_dropped_p2p_work, so an unwaited p2p handle going
        out of scope is expected to be safe.
        """
        for count, dtype in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype):
                send_tensor, recv_tensor = self._operands(count, dtype)
                if self.rank % 2 == 0:
                    dist.send(send_tensor, self.send_rank)
                    dist.recv(recv_tensor, self.recv_rank)
                else:
                    dist.recv(recv_tensor, self.recv_rank)
                    dist.send(send_tensor, self.send_rank)
                del send_tensor
                del recv_tensor
        self.sync_device()
        dist.barrier()


if __name__ == "__main__":
    run_tests()
