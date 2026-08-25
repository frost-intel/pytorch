# Owner(s): ["oncall: distributed"]
#
# Shared harness for the xccl2 integration tests.
#
# These are ports of the torchcomms XCCL integration suite
# (comms/torchcomms/tests/integration/py) onto the in-tree xccl2 c10d backend.
# The originals drive a torchcomms ``TorchComm`` object; xccl2 is reachable only
# as a c10d Backend, so the ports are written against ``torch.distributed``.
#
# Launched under torchrun, matching the original suite:
#   torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s AllReduceTest.py

import os
import tempfile
import unittest
from typing import Union

import torch
import torch.distributed as dist


BACKEND = os.environ.get("TEST_BACKEND", "xccl2")
DEVICE_TYPE = os.environ.get("TEST_DEVICE", "xpu")


def is_full_sweep() -> bool:
    """Whether to run the full parameter sweep.

    TEST_FULL_SWEEP=0 selects a reduced set of counts/dtypes/ops for a fast
    smoke test.
    """
    return os.environ.get("TEST_FULL_SWEEP", "1") == "1"


def get_rank_and_size() -> tuple[int, int]:
    """Resolve rank and world size from the launcher's environment."""
    for rank_key, size_key in (
        ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
        ("SLURM_PROCID", "SLURM_NTASKS"),
        ("PMI_RANK", "PMI_SIZE"),
        ("RANK", "WORLD_SIZE"),
    ):
        rank, size = os.environ.get(rank_key), os.environ.get(size_key)
        if rank is not None and size is not None:
            return int(rank), int(size)

    # PALS exposes no size variable of its own; fall back to whatever is set.
    pals_rank = os.environ.get("PALS_RANKID")
    pals_size = os.environ.get("WORLD_SIZE") or os.environ.get("PMI_SIZE")
    if pals_rank is not None and pals_size is not None:
        return int(pals_rank), int(pals_size)

    raise RuntimeError(
        "Could not determine rank or world size from environment variables."
    )


def get_dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.half: "Half",
        torch.float: "Float",
        torch.bfloat16: "BFloat16",
        torch.double: "Double",
        torch.int: "Int",
        torch.int8: "SignedChar",
        torch.bool: "Bool",
    }.get(dtype, "Unknown")


def get_op_name(op) -> str:
    return {
        dist.ReduceOp.SUM: "Sum",
        dist.ReduceOp.PRODUCT: "Product",
        dist.ReduceOp.MIN: "Min",
        dist.ReduceOp.MAX: "Max",
        dist.ReduceOp.BAND: "BAnd",
        dist.ReduceOp.BOR: "BOr",
        dist.ReduceOp.BXOR: "BXor",
        dist.ReduceOp.AVG: "Avg",
    }.get(op, str(op))


def filter_int8_overflow_cases(test_cases, num_ranks: int, max_ranks: int):
    """Drop int8 SUM/AVG cases whose accumulation would overflow int8."""
    if num_ranks <= max_ranks:
        return test_cases
    return [
        case
        for case in test_cases
        if not (
            case[1] == torch.int8
            and case[2] in (dist.ReduceOp.SUM, dist.ReduceOp.AVG)
        )
    ]


def verify_tensor_equality(
    output: torch.Tensor,
    expected: Union[torch.Tensor, int, float],
    description: str = "",
) -> None:
    """Compare against an expected tensor or a fill value, tolerantly for float."""
    if output.numel() == 0:
        return

    output_cpu = output.cpu()
    if isinstance(expected, (int, float)):
        expected_cpu = torch.full_like(output_cpu, float(expected))
    else:
        expected_cpu = expected.cpu()

    assert output_cpu.size() == expected_cpu.size(), (
        f"Tensor shapes don't match for {description}: "
        f"{output_cpu.size()} vs {expected_cpu.size()}"
    )
    assert output_cpu.dtype == expected_cpu.dtype, (
        f"Tensor dtypes don't match for {description}: "
        f"{output_cpu.dtype} vs {expected_cpu.dtype}"
    )

    if output_cpu.dtype in (torch.float, torch.double, torch.half, torch.bfloat16):
        diff = torch.abs(output_cpu.float() - expected_cpu.float())
        max_diff = diff.max().item()
        if max_diff >= 1e-5:
            indices = (diff >= 1e-5).nonzero()
            for i in range(min(10, indices.size(0))):
                flat_idx = int(indices[i][0].item())
                print(
                    f"Difference at index {flat_idx}: "
                    f"output={output_cpu.flatten()[flat_idx].item()}, "
                    f"expected={expected_cpu.flatten()[flat_idx].item()}"
                )
            raise AssertionError(
                f"Tensors are not close enough for {description} "
                f"(max diff {max_diff})"
            )
    else:
        if not torch.all(output_cpu.eq(expected_cpu)).item():
            indices = (output_cpu != expected_cpu).nonzero()
            for i in range(min(10, indices.size(0))):
                flat_idx = int(indices[i][0].item())
                print(
                    f"Difference at index {flat_idx}: "
                    f"output={output_cpu.flatten()[flat_idx].item()}, "
                    f"expected={expected_cpu.flatten()[flat_idx].item()}"
                )
            raise AssertionError(f"Tensors are not equal for {description}")


def parity_gap(item: str, reason: str):
    """Skip a test that exercises a torchcomms XCCL feature xccl2 lacks.

    ``item`` names the tracking entry so the skip can be removed alongside the
    commit that fills the gap.
    """
    return unittest.skip(f"xccl2 parity gap [{item}]: {reason}")


def dynamo_gap(item: str, reason: str):
    """Skip a test blocked by a dynamo limitation rather than by xccl2.

    Kept distinct from ``parity_gap`` so these do not read as backend gaps:
    they fail identically on every c10d backend.
    """
    return unittest.skip(f"dynamo limitation [{item}]: {reason}")


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", get_rank_and_size()[0]))


def make_store(name: str):
    """A FileStore isolated to ``name``, for tests that build their own group.

    The originals allocate a TCPStore on a free port and publish it through a
    side file. The suite runs single-node under torchrun, so a FileStore on a
    path all ranks can derive is equivalent and has no port to leak. The run id
    keeps concurrent invocations from colliding.
    """
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getppid()))
    path = os.path.join(
        tempfile.gettempdir(), f"xccl2_integration_{run_id}_{name}"
    )
    return dist.FileStore(path, get_rank_and_size()[1])


class Xccl2TestBase(unittest.TestCase):
    """Base for the ported integration tests.

    Creates one process group for the whole class, mirroring the original
    suite's per-test ``TorchCommTestWrapper``. xccl2 binds its device lazily on
    first collective, so the device is set here rather than at init.
    """

    @classmethod
    def setUpClass(cls) -> None:
        rank, world_size = get_rank_and_size()
        if not dist.is_initialized():
            dist.init_process_group(
                backend=BACKEND, rank=rank, world_size=world_size
            )
        cls.rank = dist.get_rank()
        cls.num_ranks = dist.get_world_size()
        if DEVICE_TYPE == "xpu":
            device_index = cls.rank % torch.xpu.device_count()
            torch.xpu.set_device(device_index)
            cls.device = torch.device("xpu", device_index)
        else:
            cls.device = torch.device(DEVICE_TYPE)

    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def wait(self, work) -> None:
        """Wait on a Work handle, tolerating the sync case.

        torchcomms returns a work object for both sync and async calls; c10d
        returns None when async_op=False.
        """
        if work is not None:
            work.wait()

    def sync_device(self) -> None:
        if self.device.type == "xpu":
            torch.xpu.synchronize()


class CollectiveVariantsMixin:
    """The call-shape variants every collective test in the original suite repeats.

    Each torchcomms collective test defines the same five bodies -- sync with a
    work handle, sync discarding it, async, async then dropping the handle, and
    input dropped right after enqueue -- differing only in how operands are
    built, which collective is called, and how the result is checked. Those
    three are the subclass's job:

        get_test_cases() -> iterable of case tuples
        make_operands(case) -> object passed back to call()/verify()
        call(operands, case, async_op) -> Work or None
        verify(operands, case) -> None

    The original's CUDA-graph variants are ncclx-only and are not ported.
    """

    def _cases(self):
        return list(self.get_test_cases())

    def test_sync(self):
        """Synchronous call returns no handle and completes before returning."""
        for case in self._cases():
            with self.subTest(case=case):
                operands = self.make_operands(case)
                work = self.call(operands, case, async_op=False)
                self.assertIsNone(work)
                self.verify(operands, case)

    def test_sync_no_work(self):
        """Synchronous call with the return value discarded."""
        for case in self._cases():
            with self.subTest(case=case):
                operands = self.make_operands(case)
                self.call(operands, case, async_op=False)
                self.verify(operands, case)

    def test_async(self):
        """Asynchronous call waited to completion."""
        for case in self._cases():
            with self.subTest(case=case):
                operands = self.make_operands(case)
                work = self.call(operands, case, async_op=True)
                work.wait()
                self.verify(operands, case)

    def test_async_early_reset(self):
        """Dropping the work handle after waiting must not affect the result."""
        for case in self._cases():
            with self.subTest(case=case):
                operands = self.make_operands(case)
                work = self.call(operands, case, async_op=True)
                work.wait()
                work = None
                self.verify(operands, case)

    def test_input_deleted(self):
        """Dropping operands right after enqueue must not crash the watchdog.

        Results are unverifiable once the operands are gone; this asserts only
        that the backend keeps the buffers alive until the op retires.
        """
        for case in self._cases():
            with self.subTest(case=case):
                operands = self.make_operands(case)
                self.call(operands, case, async_op=False)
                del operands
        self.sync_device()
        dist.barrier()


def run_tests() -> None:
    unittest.main(verbosity=2)
