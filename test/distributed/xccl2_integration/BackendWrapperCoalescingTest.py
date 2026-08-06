# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms BackendWrapperCoalescingTest.py onto xccl2.
#
# The original checks that BackendWrapper implements c10d's coalescing hooks so
# batch_isend_irecv takes the grouped path. xccl2 sets supportsCoalescing()
# true and implements startCoalescing/endCoalescing directly, so the same
# assertions apply with the wrapper removed from the path.
#
# The mixed send+recv pattern below is the reason coalescing matters: issued
# ungrouped on one stream, a bidirectional ring deadlocks.

import torch
import torch.distributed as dist

from common import run_tests, Xccl2TestBase


class CoalescingTest(Xccl2TestBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        torch.set_default_device(cls.device)

    def test_supports_coalescing_is_true(self):
        """c10d's _coalescing_manager only groups if the backend opts in."""
        pg = dist.distributed_c10d._get_default_group()
        backend = pg._get_backend(self.device)
        self.assertTrue(
            backend.supports_coalescing,
            "xccl2 supports_coalescing must be True so batch_isend_irecv "
            "takes the coalescing path",
        )

    def test_batch_isend_irecv_mixed_send_recv(self):
        """Mixed isend+irecv in one batch; deadlocks if not grouped."""
        if self.num_ranks < 2:
            self.skipTest("need at least 2 ranks for batch_isend_irecv")

        peer = (self.rank + 1) % self.num_ranks
        recv_peer = (self.rank - 1) % self.num_ranks
        send_tensor = torch.full((4,), float(self.rank), dtype=torch.float32)
        recv_tensor = torch.empty(4, dtype=torch.float32)

        ops = [
            dist.P2POp(dist.isend, send_tensor, peer),
            dist.P2POp(dist.irecv, recv_tensor, recv_peer),
        ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        self.sync_device()

        expected = torch.full((4,), float(recv_peer), dtype=torch.float32)
        self.assertTrue(
            torch.equal(recv_tensor, expected),
            f"recv mismatch: got {recv_tensor.tolist()}, "
            f"expected {expected.tolist()}",
        )

    def test_batch_isend_irecv_multiple_peers(self):
        """N sends + N recvs across several peers in one coalesced batch."""
        if self.num_ranks < 3:
            self.skipTest("need at least 3 ranks for multi-peer batch")

        send_tensors = [
            torch.full((4,), float(self.rank * 10 + i), dtype=torch.float32)
            for i in range(2)
        ]
        recv_tensors = [torch.empty(4, dtype=torch.float32) for _ in range(2)]
        send_peers = [(self.rank + 1) % self.num_ranks, (self.rank + 2) % self.num_ranks]
        recv_peers = [(self.rank - 1) % self.num_ranks, (self.rank - 2) % self.num_ranks]

        ops = []
        for i in range(2):
            ops.append(dist.P2POp(dist.isend, send_tensors[i], send_peers[i]))
            ops.append(dist.P2POp(dist.irecv, recv_tensors[i], recv_peers[i]))

        for req in dist.batch_isend_irecv(ops):
            req.wait()
        self.sync_device()

        for i in range(2):
            expected = torch.full(
                (4,), float(recv_peers[i] * 10 + i), dtype=torch.float32
            )
            self.assertTrue(
                torch.equal(recv_tensors[i], expected),
                f"slot {i} (from rank {recv_peers[i]}): got "
                f"{recv_tensors[i].tolist()}, expected {expected.tolist()}",
            )


if __name__ == "__main__":
    run_tests()
