# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms C10dBatchIsendIrecvTest.py onto xccl2.
#
# The original guards the BackendWrapper coalescing path that pipeline
# parallel depends on. xccl2 implements startCoalescing/endCoalescing itself,
# so the port drops `dist.config.use_torchcomms = True` and keeps the patterns
# unchanged: each is a shape PP actually issues, and each deadlocks if the
# batch is not grouped into a single oneCCL group.

import torch
import torch.distributed as dist

from common import run_tests, Xccl2TestBase


class BatchIsendIrecvTest(Xccl2TestBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        torch.set_default_device(cls.device)

    def setUp(self):
        if self.num_ranks < 2:
            self.skipTest("batch_isend_irecv tests require world_size >= 2")
        self.next = (self.rank + 1) % self.num_ranks
        self.prev = (self.rank - 1 + self.num_ranks) % self.num_ranks

    def test_supports_coalescing(self):
        """batch_isend_irecv only groups if the backend advertises support."""
        backend = dist.group.WORLD._get_backend(self.device)
        self.assertTrue(backend.supports_coalescing)

    def test_batch_isend_irecv_mixed_ring(self):
        """Send to next while receiving from prev: the PP 1F1B middle stage."""
        send_tensor = torch.tensor([self.rank], dtype=torch.float32)
        recv_tensor = torch.empty(1, dtype=torch.float32)

        ops = [
            dist.P2POp(dist.isend, send_tensor, self.next),
            dist.P2POp(dist.irecv, recv_tensor, self.prev),
        ]
        for w in dist.batch_isend_irecv(ops):
            w.wait()

        self.assertEqual(recv_tensor.item(), float(self.prev))

    def test_batch_isend_irecv_recv_first(self):
        """Order within a coalesced batch must not matter."""
        send_tensor = torch.tensor([self.rank * 10 + 1], dtype=torch.float32)
        recv_tensor = torch.empty(1, dtype=torch.float32)

        ops = [
            dist.P2POp(dist.irecv, recv_tensor, self.prev),
            dist.P2POp(dist.isend, send_tensor, self.next),
        ]
        for w in dist.batch_isend_irecv(ops):
            w.wait()

        self.assertEqual(recv_tensor.item(), float(self.prev * 10 + 1))

    def test_batch_isend_irecv_multiple_ops_per_peer(self):
        """Several ops to the same peers in one batch, as when activations
        and gradients overlap on one neighbour."""
        send1 = torch.tensor([self.rank], dtype=torch.float32)
        send2 = torch.tensor([self.rank + 100], dtype=torch.float32)
        recv1 = torch.empty(1, dtype=torch.float32)
        recv2 = torch.empty(1, dtype=torch.float32)

        ops = [
            dist.P2POp(dist.isend, send1, self.next),
            dist.P2POp(dist.irecv, recv1, self.prev),
            dist.P2POp(dist.isend, send2, self.next),
            dist.P2POp(dist.irecv, recv2, self.prev),
        ]
        for w in dist.batch_isend_irecv(ops):
            w.wait()

        self.assertEqual(recv1.item(), float(self.prev))
        self.assertEqual(recv2.item(), float(self.prev + 100))

    def test_individual_isend_irecv_outside_coalescing(self):
        """Ordinary send/recv after a batch must not see leftover state."""
        send_tensor = torch.tensor([self.rank], dtype=torch.float32)
        recv_tensor = torch.empty(1, dtype=torch.float32)
        ops = [
            dist.P2POp(dist.isend, send_tensor, self.next),
            dist.P2POp(dist.irecv, recv_tensor, self.prev),
        ]
        for w in dist.batch_isend_irecv(ops):
            w.wait()

        send2 = torch.tensor([self.rank + 1000], dtype=torch.float32)
        recv2 = torch.empty(1, dtype=torch.float32)
        if self.rank % 2 == 0:
            dist.send(send2, dst=self.next)
            dist.recv(recv2, src=self.prev)
        else:
            dist.recv(recv2, src=self.prev)
            dist.send(send2, dst=self.next)
        self.assertEqual(recv2.item(), float(self.prev + 1000))


if __name__ == "__main__":
    run_tests()
