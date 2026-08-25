# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms MultiCommTest.py onto xccl2.
#
# The original sweeps a 3x3 matrix: {two, three, mixed-ops} communicators x
# {separate stores, no store, mixed store}. The store axis is a torchcomms
# bootstrap concern -- each TorchComm is created from a store the test hands
# it. c10d derives every subgroup's store from the default group's by prefixing
# it, so the axis has no counterpart here and only the communicator-count and
# op-mix axes survive.
#
# What the original is really guarding is that several communicators over the
# same ranks stay independent and can be driven concurrently without their
# operations interleaving on the wire. That is preserved.

import torch
import torch.distributed as dist

from common import run_tests, verify_tensor_equality, Xccl2TestBase


class MultiCommTest(Xccl2TestBase):
    def _make_groups(self, count):
        """`count` concurrent groups spanning every rank."""
        return [dist.new_group(ranks=list(range(self.num_ranks))) for _ in range(count)]

    def _destroy(self, groups):
        for group in groups:
            dist.destroy_process_group(group)

    def _verify_communication(self, group):
        """One all_reduce on a single group."""
        tensor = torch.ones(10, dtype=torch.float, device=self.device) * float(
            self.rank + 1
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        expected = self.num_ranks * (self.num_ranks + 1) / 2
        verify_tensor_equality(tensor, expected, "single-group all_reduce")

    def _verify_simultaneous_communication(self, groups):
        """Issue on every group before waiting on any."""
        tensors = [
            torch.ones(10, dtype=torch.float, device=self.device)
            * float(self.rank + 1)
            for _ in groups
        ]
        works = [
            dist.all_reduce(
                tensors[i], op=dist.ReduceOp.SUM, group=group, async_op=True
            )
            for i, group in enumerate(groups)
        ]
        for work in works:
            work.wait()

        expected = self.num_ranks * (self.num_ranks + 1) / 2
        for i, tensor in enumerate(tensors):
            verify_tensor_equality(
                tensor, expected, f"group_{i} simultaneous all_reduce result"
            )

    def test_two_comms(self):
        groups = self._make_groups(2)
        try:
            for group in groups:
                self._verify_communication(group)
            self._verify_simultaneous_communication(groups)
        finally:
            self._destroy(groups)

    def test_three_comms(self):
        groups = self._make_groups(3)
        try:
            for group in groups:
                self._verify_communication(group)
            self._verify_simultaneous_communication(groups)
        finally:
            self._destroy(groups)

    def test_mixed_ops(self):
        """A different collective on each group, all in flight together."""
        groups = self._make_groups(3)
        try:
            all_reduce_t = torch.ones(
                10, dtype=torch.float, device=self.device
            ) * float(self.rank + 1)
            broadcast_t = torch.ones(
                10, dtype=torch.float, device=self.device
            ) * float(self.rank + 1)
            gather_out = [
                torch.empty(10, dtype=torch.float, device=self.device)
                for _ in range(self.num_ranks)
            ]
            gather_in = torch.ones(
                10, dtype=torch.float, device=self.device
            ) * float(self.rank + 1)

            works = [
                dist.all_reduce(
                    all_reduce_t,
                    op=dist.ReduceOp.SUM,
                    group=groups[0],
                    async_op=True,
                ),
                dist.broadcast(broadcast_t, src=0, group=groups[1], async_op=True),
                dist.all_gather(
                    gather_out, gather_in, group=groups[2], async_op=True
                ),
            ]
            for work in works:
                work.wait()

            verify_tensor_equality(
                all_reduce_t,
                self.num_ranks * (self.num_ranks + 1) / 2,
                "mixed all_reduce",
            )
            verify_tensor_equality(broadcast_t, 1.0, "mixed broadcast")
            for r in range(self.num_ranks):
                verify_tensor_equality(
                    gather_out[r], float(r + 1), f"mixed all_gather slot {r}"
                )
        finally:
            self._destroy(groups)

    def test_groups_are_independent(self):
        """A collective on one group must not disturb another's state."""
        groups = self._make_groups(2)
        try:
            a = torch.ones(10, dtype=torch.float, device=self.device)
            dist.all_reduce(a, op=dist.ReduceOp.SUM, group=groups[0])
            verify_tensor_equality(a, float(self.num_ranks), "group 0")

            b = torch.ones(10, dtype=torch.float, device=self.device) * 2.0
            dist.all_reduce(b, op=dist.ReduceOp.SUM, group=groups[1])
            verify_tensor_equality(b, float(2 * self.num_ranks), "group 1")

            dist.all_reduce(a, op=dist.ReduceOp.SUM, group=groups[0])
            verify_tensor_equality(
                a, float(self.num_ranks * self.num_ranks), "group 0 reused"
            )
        finally:
            self._destroy(groups)


if __name__ == "__main__":
    run_tests()
