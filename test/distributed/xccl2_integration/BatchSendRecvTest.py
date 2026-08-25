# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms BatchSendRecvTest.py onto the xccl2 c10d backend.
# torchcomms' batch_op_create()/batch_op_issue() map to dist.batch_isend_irecv
# over a list of dist.P2POp. xccl2 routes this through BatchSendRecv, which
# issues every send before any recv to avoid an IPC deadlock on XPU.

import itertools

import torch
import torch.distributed as dist

from common import is_full_sweep, run_tests, Xccl2TestBase


class BatchSendRecvTest(Xccl2TestBase):
    counts = [4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]

    def get_test_cases(self):
        return list(itertools.product(self.counts, self.dtypes))

    @property
    def send_rank(self):
        return (self.rank + 1) % self.num_ranks

    @property
    def recv_rank(self):
        return (self.rank - 1) % self.num_ranks

    def _ops(self, count, dtype):
        send_tensor = torch.ones(count, dtype=dtype, device=self.device) * (
            self.rank + 1
        )
        recv_tensor = torch.zeros(count, dtype=dtype, device=self.device)
        ops = [
            dist.P2POp(dist.isend, send_tensor, self.send_rank),
            dist.P2POp(dist.irecv, recv_tensor, self.recv_rank),
        ]
        return ops, send_tensor, recv_tensor

    def _verify(self, recv_tensor):
        torch.testing.assert_close(
            recv_tensor.cpu(),
            torch.full_like(recv_tensor.cpu(), self.recv_rank + 1),
            msg=f"batch recv on rank {self.rank} from rank {self.recv_rank}",
        )

    def test_batch_sendrecv(self):
        """A batched send/recv pair around the ring."""
        for count, dtype in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype):
                ops, _, recv_tensor = self._ops(count, dtype)
                for work in dist.batch_isend_irecv(ops):
                    work.wait()
                self._verify(recv_tensor)

    def test_batch_sendrecv_early_reset(self):
        """Dropping the handles after waiting must not affect the result."""
        for count, dtype in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype):
                ops, _, recv_tensor = self._ops(count, dtype)
                works = dist.batch_isend_irecv(ops)
                for work in works:
                    work.wait()
                works = None
                self._verify(recv_tensor)

    def test_batch_sendrecv_all_peers(self):
        """A full exchange with every peer in one batch."""
        count, dtype = 1024, torch.float
        send_tensors = [
            torch.ones(count, dtype=dtype, device=self.device) * (self.rank + 1)
            for _ in range(self.num_ranks)
        ]
        recv_tensors = [
            torch.zeros(count, dtype=dtype, device=self.device)
            for _ in range(self.num_ranks)
        ]
        ops = []
        for peer in range(self.num_ranks):
            if peer == self.rank:
                continue
            ops.append(dist.P2POp(dist.isend, send_tensors[peer], peer))
            ops.append(dist.P2POp(dist.irecv, recv_tensors[peer], peer))

        for work in dist.batch_isend_irecv(ops):
            work.wait()

        for peer in range(self.num_ranks):
            if peer == self.rank:
                continue
            torch.testing.assert_close(
                recv_tensors[peer].cpu(),
                torch.full_like(recv_tensors[peer].cpu(), peer + 1),
                msg=f"batch recv from peer {peer}",
            )

    def test_batch_sendrecv_input_deleted(self):
        """Dropping the buffers right after enqueue must not crash."""
        for count, dtype in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype):
                ops, send_tensor, recv_tensor = self._ops(count, dtype)
                for work in dist.batch_isend_irecv(ops):
                    work.wait()
                del ops
                del send_tensor
                del recv_tensor
        self.sync_device()
        dist.barrier()


if __name__ == "__main__":
    run_tests()
