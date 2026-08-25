# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms SplitTest.py onto the xccl2 c10d backend.
#
# torchcomms exposes TorchComm.split() to carve a sub-communicator out of an
# existing one. The c10d equivalent is dist.split_group(), which reaches
# ProcessGroupXCCL::split and so onecclCommSplit via XcclApi::commSplit.
#
# Unlike torchcomms' split(), where each rank names only its own membership,
# split_group() is collective over the parent: every rank passes the same
# split_ranks and receives back whichever subgroup it belongs to.

import torch
import torch.distributed as dist

from common import (
    BACKEND,
    DEVICE_TYPE,
    get_rank_and_size,
    run_tests,
    Xccl2TestBase,
)


class SplitTest(Xccl2TestBase):
    @classmethod
    def setUpClass(cls) -> None:
        # split_group() refuses to run unless the default group is bound to a
        # device, which the shared harness does not do.
        if not dist.is_initialized():
            rank, world_size = get_rank_and_size()
            device_id = None
            if DEVICE_TYPE == "xpu":
                index = rank % torch.xpu.device_count()
                torch.xpu.set_device(index)
                device_id = torch.device("xpu", index)
            dist.init_process_group(
                backend=BACKEND,
                rank=rank,
                world_size=world_size,
                device_id=device_id,
            )
        super().setUpClass()

    def _halves(self):
        """Split the world into even-rank and odd-rank groups."""
        evens = [r for r in range(self.num_ranks) if r % 2 == 0]
        odds = [r for r in range(self.num_ranks) if r % 2 == 1]
        return evens, odds

    def _split_halves(self):
        """Split the parent into even/odd halves and hand back this rank's."""
        evens, odds = self._halves()
        # split_group() documents that the parent must already be initialized.
        dist.barrier()
        subgroup = dist.split_group(split_ranks=[evens, odds])
        return subgroup, (evens if self.rank % 2 == 0 else odds)

    def test_split_produces_working_subgroup(self):
        subgroup, my_group_ranks = self._split_halves()
        self.assertEqual(dist.get_world_size(group=subgroup), len(my_group_ranks))

        tensor = torch.ones(1024, device=self.device) * (self.rank + 1)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=subgroup)

        expected = sum(r + 1 for r in my_group_ranks)
        torch.testing.assert_close(
            tensor.cpu(), torch.full_like(tensor.cpu(), expected)
        )
        dist.destroy_process_group(subgroup)

    def test_split_group_ranks_are_renumbered(self):
        subgroup, my_group_ranks = self._split_halves()
        self.assertEqual(
            dist.get_rank(group=subgroup), my_group_ranks.index(self.rank)
        )
        dist.destroy_process_group(subgroup)

    def test_parent_group_still_usable_after_split(self):
        subgroup, _ = self._split_halves()
        dist.destroy_process_group(subgroup)

        tensor = torch.ones(1024, device=self.device) * (self.rank + 1)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        n = self.num_ranks
        torch.testing.assert_close(
            tensor.cpu(), torch.full_like(tensor.cpu(), n * (n + 1) / 2)
        )


if __name__ == "__main__":
    run_tests()
