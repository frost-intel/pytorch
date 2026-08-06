# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms BackendWrapperShutdownTest.py onto xccl2.
#
# The original checks that BackendWrapper::shutdown/abort close the underlying
# TorchComm exactly once, since two wrappers can share one comm in a mixed
# "cpu:gloo,<dev>:<backend>" group. xccl2 has no wrapper, but the same hazard
# exists at the c10d layer: destroy_process_group() tears down every
# sub-backend, and ProcessGroupXCCL::shutdown must tolerate being reached twice
# without raising "already finalized".
#
# Each test builds and destroys its own group -- the init/destroy cycle is what
# is under test, so the shared class-level group of the other files is unused.


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


class ShutdownTest(Xccl2TestBase):
    @classmethod
    def setUpClass(cls) -> None:
        # Deliberately no group here; each test owns its lifecycle.
        cls.rank, cls.num_ranks = get_rank_and_size()

    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def _init_pg(self, store_name: str) -> torch.device:
        """Bind the device before init so each rank gets its own communicator."""
        index = local_rank()
        if DEVICE_TYPE == "xpu":
            torch.xpu.set_device(index)
        device = torch.device(DEVICE_TYPE, index)
        dist.init_process_group(
            backend=BACKEND,
            store=make_store(store_name),
            rank=self.rank,
            world_size=self.num_ranks,
        )
        torch.set_default_device(device)
        return device

    def test_destroy_after_collective_no_hang(self):
        """init -> all_reduce -> destroy must finish rather than hang."""
        self._init_pg("destroy_after_collective_no_hang")
        try:
            tensor = torch.ones(8, dtype=torch.float32)
            dist.all_reduce(tensor)
            self.assertEqual(tensor[0].item(), float(self.num_ranks))
        finally:
            dist.destroy_process_group()

    def test_mixed_backend_destroy_idempotent(self):
        """A cpu:gloo,xpu:xccl2 group tears down both sub-backends cleanly."""
        index = local_rank()
        if DEVICE_TYPE == "xpu":
            torch.xpu.set_device(index)
        local_device = f"{DEVICE_TYPE}:{index}"
        dist.init_process_group(
            backend=f"cpu:gloo,{DEVICE_TYPE}:{BACKEND}",
            store=make_store("mixed_backend_destroy_idempotent"),
            rank=self.rank,
            world_size=self.num_ranks,
            device_id=torch.device(local_device),
        )
        try:
            torch.set_default_device(local_device)
            cpu_tensor = torch.ones(4, dtype=torch.float32, device="cpu")
            dev_tensor = torch.ones(4, dtype=torch.float32, device=local_device)
            dist.all_reduce(cpu_tensor)
            dist.all_reduce(dev_tensor)
            self.assertEqual(cpu_tensor[0].item(), float(self.num_ranks))
            self.assertEqual(dev_tensor[0].item(), float(self.num_ranks))
        finally:
            dist.destroy_process_group()


# abort() is deliberately untested: ProcessGroupXCCL::abortXcclComm calls
# ::abort() whenever abort_process_on_timeout_or_error_ is set, and that member
# is hardcoded true with no setter, so any in-process test of it kills the
# runner. The original omits it for the same reason.
#
# Idempotent destroy is likewise only tested through the mixed-backend case
# above: c10d rejects a repeated destroy at the Python layer ("Invalid process
# group specified"), so the shared-comm double-shutdown the original cares
# about is only reachable via a group with two sub-backends.


if __name__ == "__main__":
    run_tests()
