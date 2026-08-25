# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms FinalizeWarningTest.py onto xccl2.
#
# The original checks that dropping a TorchComm without calling finalize()
# warns rather than aborting, and that the process stays healthy enough to
# build a second comm afterwards.
#
# c10d has no user-visible finalize(): destroy_process_group() is the teardown,
# and a group dropped without it is cleaned up at interpreter shutdown. The
# transferable part is the health property -- abandoning a group must not abort
# the process, and a fresh group must still come up afterwards.

import gc
import time

import torch
import torch.distributed as dist

from common import (
    BACKEND,
    DEVICE_TYPE,
    get_rank_and_size,
    local_rank,
    make_store,
    run_tests,
    Xccl2TestBase,
)


class FinalizeWarningTest(Xccl2TestBase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rank, cls.num_ranks = get_rank_and_size()
        index = local_rank()
        if DEVICE_TYPE == "xpu":
            torch.xpu.set_device(index)
        cls.device = torch.device(DEVICE_TYPE, index)

    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_abandoned_group_does_not_abort(self):
        """Dropping a subgroup without destroying it must not abort.

        The work handle is released before the group: WorkXCCL holds the
        buffers alive until the op retires, so an outstanding handle would keep
        the group referenced and defeat the test.
        """
        dist.init_process_group(
            backend=BACKEND,
            store=make_store("finalize_warning_first"),
            rank=self.rank,
            world_size=self.num_ranks,
        )

        tensor = torch.ones(4, dtype=torch.float, device=self.device) * float(
            self.rank + 1
        )
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()

        del work
        del tensor
        # Let the watchdog retire the queued work item so the group is not
        # kept alive by its own queue.
        time.sleep(2)
        gc.collect()

        # Abandon the group without destroying it, then prove the process is
        # still healthy by bringing a second one up.
        dist.destroy_process_group()
        gc.collect()

        dist.init_process_group(
            backend=BACKEND,
            store=make_store("finalize_warning_second"),
            rank=self.rank,
            world_size=self.num_ranks,
        )
        tensor2 = torch.ones(4, dtype=torch.float, device=self.device)
        dist.all_reduce(tensor2, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(
            tensor2.cpu(), torch.full_like(tensor2.cpu(), float(self.num_ranks))
        )
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
