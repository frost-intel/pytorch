# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms BackendWrapperAllGatherAliasedTest.py onto xccl2.
#
# The original exercises BackendWrapper, the shim that dresses a torchcomms
# TorchComm up as a c10d::Backend, so it is already written against plain
# torch.distributed and only needs the `dist.config.use_torchcomms = True`
# opt-in dropped: xccl2 is a c10d::Backend natively, with no wrapper in the
# path. What is under test survives the port unchanged -- c10d permits the
# per-rank output tensors of all_gather to alias one buffer, and the backend
# must not race writes into it.

import torch
import torch.distributed as dist

from common import run_tests, Xccl2TestBase


class AllGatherAliasedTest(Xccl2TestBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        torch.set_default_device(cls.device)

    def test_distinct_outputs_fast_path(self):
        """Distinct output buffers gather the per-rank inputs in rank order."""
        input_tensor = torch.tensor([float(self.rank)], dtype=torch.float32)
        output_list = [
            torch.empty(1, dtype=torch.float32) for _ in range(self.num_ranks)
        ]

        dist.all_gather(output_list, input_tensor)

        for r in range(self.num_ranks):
            self.assertEqual(
                output_list[r].item(),
                float(r),
                f"slot {r}: expected {float(r)}, got {output_list[r].item()}",
            )

    def test_aliased_outputs_no_crash(self):
        """Outputs all aliasing one buffer must not race; last rank wins.

        xccl2's all_gather flattens into a staging tensor and copies each
        slice back in rank order, so the final value is the last rank's --
        the same observable behaviour as stock ProcessGroupNCCL.
        """
        input_tensor = torch.tensor([float(self.rank)], dtype=torch.float32)
        shared = torch.full((1,), -1.0, dtype=torch.float32)
        output_list = [shared for _ in range(self.num_ranks)]

        dist.all_gather(output_list, input_tensor)

        self.assertNotEqual(
            shared.item(),
            -1.0,
            "shared aliased buffer was never written - gather skipped",
        )
        self.assertEqual(
            shared.item(),
            float(self.num_ranks - 1),
            f"expected last-rank-wins value {float(self.num_ranks - 1)}, "
            f"got {shared.item()}",
        )

    def test_aliased_then_distinct_does_not_pollute(self):
        """The staging buffer used by the aliased path must not leak."""
        shared = torch.zeros(1, dtype=torch.float32)
        dist.all_gather(
            [shared for _ in range(self.num_ranks)],
            torch.tensor([float(self.rank)], dtype=torch.float32),
        )

        output_list = [
            torch.empty(1, dtype=torch.float32) for _ in range(self.num_ranks)
        ]
        dist.all_gather(
            output_list,
            torch.tensor([float(self.rank) + 100.0], dtype=torch.float32),
        )

        for r in range(self.num_ranks):
            self.assertEqual(output_list[r].item(), float(r) + 100.0)


if __name__ == "__main__":
    run_tests()
