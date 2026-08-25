# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms AllToAllvSingleTest.py onto the xccl2 c10d backend.
# torchcomms' all_to_all_v_single(out, in, out_splits, in_splits, async_op)
# maps to dist.all_to_all_single with explicit split sizes.
#
# Each (src, dst) pair carries a distinct fill value so a misplaced section is
# caught rather than masked, and the output buffer is pre-filled with a
# sentinel so an unwritten section fails instead of reading as a valid zero.

import itertools
from enum import Enum

import torch
import torch.distributed as dist

from common import (
    CollectiveVariantsMixin,
    get_dtype_name,
    is_full_sweep,
    run_tests,
    Xccl2TestBase,
)


class SizePattern(Enum):
    UNIFORM = "Uniform"
    VARIABLE = "Variable"
    ZERO_SIZES = "ZeroSizes"
    ALL_ZERO = "AllZero"


def fill_value(dtype, src, dst):
    """Distinct value for the section rank ``src`` sends to rank ``dst``."""
    if dtype == torch.int8:
        return (src * 10 + dst + 1) % 128
    return src * 100 + dst + 1


class AllToAllvSingleTest(CollectiveVariantsMixin, Xccl2TestBase):
    counts = [4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = (
        [torch.float, torch.bfloat16, torch.int, torch.int8]
        if is_full_sweep()
        else [torch.float, torch.bfloat16]
    )

    def get_test_cases(self):
        cases = list(itertools.product(self.counts, SizePattern, self.dtypes))
        # AllZero does not vary with count; keep a single representative case.
        return [
            case
            for case in cases
            if case[1] != SizePattern.ALL_ZERO or case[0] == self.counts[0]
        ]

    def _splits(self, pattern, count):
        """Return (input_split_sizes, output_split_sizes) for this rank."""
        n = self.num_ranks
        if pattern == SizePattern.UNIFORM:
            return [count] * n, [count] * n
        if pattern == SizePattern.VARIABLE:
            # Rank i sends (i+1)*(j+1)*count to rank j.
            return (
                [(self.rank + 1) * (j + 1) * count for j in range(n)],
                [(i + 1) * (self.rank + 1) * count for i in range(n)],
            )
        if pattern == SizePattern.ZERO_SIZES:
            # Symmetric: if this rank sends 0 to i, i sends 0 back.
            sizes = [0 if (self.rank + i) % 3 == 0 else count for i in range(n)]
            return sizes, list(sizes)
        return [0] * n, [0] * n

    def make_operands(self, case):
        count, pattern, dtype = case
        in_splits, out_splits = self._splits(pattern, count)

        input_tensor = torch.zeros(sum(in_splits), dtype=dtype, device=self.device)
        offset = 0
        for dst in range(self.num_ranks):
            if in_splits[dst] > 0:
                input_tensor[offset : offset + in_splits[dst]].fill_(
                    fill_value(dtype, self.rank, dst)
                )
            offset += in_splits[dst]

        sentinel = float("nan") if dtype.is_floating_point else -1
        output_tensor = torch.full(
            (sum(out_splits),), sentinel, dtype=dtype, device=self.device
        )
        return output_tensor, input_tensor, out_splits, in_splits

    def call(self, operands, case, async_op):
        output_tensor, input_tensor, out_splits, in_splits = operands
        return dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=out_splits,
            input_split_sizes=in_splits,
            async_op=async_op,
        )

    def verify(self, operands, case):
        _, _, dtype = case
        output_tensor, _, out_splits, _ = operands
        output_cpu = output_tensor.cpu()
        offset = 0
        for src in range(self.num_ranks):
            size = out_splits[src]
            if size > 0:
                section = output_cpu[offset : offset + size]
                expected = torch.full(
                    (size,), fill_value(dtype, src, self.rank), dtype=dtype
                )
                torch.testing.assert_close(
                    section,
                    expected,
                    msg=(
                        f"all_to_all_v_single section from rank {src} "
                        f"({get_dtype_name(dtype)})"
                    ),
                )
            offset += size


if __name__ == "__main__":
    run_tests()
