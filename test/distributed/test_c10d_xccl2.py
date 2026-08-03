# Owner(s): ["oncall: distributed"]
#
# Tests specific to the in-tree torchcomms XCCL backend.

import time

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_xccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_XPU


class ProcessGroupXCCL2Test(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str:
        return "xccl2"

    @classmethod
    def device_type(cls) -> str:
        return "xpu"

    @property
    def device(self) -> torch.device:
        return torch.device("xpu", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.xpu.set_device(self.rank)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_watchdog_does_not_release_python_backed_tensor(self) -> None:
        class TensorSubclass(torch.Tensor):
            pass

        tensor = torch.ones(4, device=self.device).as_subclass(TensorSubclass)
        outputs = [torch.empty(4, device=self.device) for _ in range(self.world_size)]
        work = dist.all_gather(outputs, tensor, async_op=True)
        del tensor
        del work

        torch.xpu.synchronize()
        time.sleep(2)
        dist.barrier()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_options_not_shared_with_legacy_backend(self) -> None:
        # nccl2 reuses core's ProcessGroupNCCL.Options, but the legacy XCCL
        # backend lives in torch-xpu-ops and carries an unrelated Options, so
        # xccl2 binds its own. Guard against the two being conflated.
        self.assertIsNot(
            dist.ProcessGroupXCCL2.Options, dist.ProcessGroupXCCL.Options
        )
        opts = dist.ProcessGroupXCCL2.Options()
        self.assertFalse(opts.is_high_priority_stream)
        self.assertEqual(opts.hints, {})
        opts.is_high_priority_stream = True
        opts.hints = {"key": "value"}
        self.assertTrue(opts.is_high_priority_stream)
        self.assertEqual(opts.hints, {"key": "value"})

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_backend_identity(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        self.assertIsInstance(backend, dist.ProcessGroupXCCL2)
        self.assertEqual(dist.get_backend(), "xccl2")

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_backend_version_reported_after_bootstrap(self) -> None:
        # backend_version_ is filled in by initXcclResources, so force the lazy
        # bootstrap before reading it.
        dist.all_reduce(torch.ones(1, device=self.device))
        backend = dist.get_backend_impl(device=self.device)
        self.assertTrue(backend.backend_version.isdigit())


class _ProcessGroupXCCL2OptionsTest(MultiProcContinuousTest):
    """Base for groups initialized with backend specific options."""

    @classmethod
    def backend_str(cls) -> str:
        return "xccl2"

    @classmethod
    def device_type(cls) -> str:
        return "xpu"

    @property
    def device(self) -> torch.device:
        return torch.device("xpu", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.xpu.set_device(self.rank)

    def _check_all_reduce(self) -> None:
        t = torch.full((4,), float(self.rank), device=self.device)
        dist.all_reduce(t)
        expected = float(sum(range(self.world_size)))
        self.assertEqual(t, torch.full((4,), expected, device=self.device))


class ProcessGroupXCCL2HintsTest(_ProcessGroupXCCL2OptionsTest):
    @classmethod
    def opts(cls, high_priority_stream=False):
        opts = dist.ProcessGroupXCCL2.Options()
        opts.is_high_priority_stream = True
        opts.hints = {"is_high_priority_stream": "1"}
        return opts

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_collective_with_options(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        self.assertTrue(backend.options.is_high_priority_stream)
        self.assertEqual(backend.options.hints, {"is_high_priority_stream": "1"})
        self._check_all_reduce()


if __name__ == "__main__":
    if TEST_XPU:
        run_tests()
