# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms C10dTorchCommTest.py onto xccl2.
#
# The original is already written against plain torch.distributed -- it drives
# torchcomms only through `dist.config.use_torchcomms = True`, which routes the
# c10d calls into a BackendWrapper. Dropping that line and selecting the xccl2
# backend leaves the same c10d surface under test.
#
# The original parametrizes over reduce ops with torch.testing._internal's
# `parametrize`; this uses subTest to match the rest of this suite and to keep
# the file runnable under plain pytest.

import torch
import torch.distributed as dist

from common import get_op_name, run_tests, Xccl2TestBase


class C10dCollectivesTest(Xccl2TestBase):
    REDUCE_OPS = [
        dist.ReduceOp.SUM,
        dist.ReduceOp.AVG,
        dist.ReduceOp.MIN,
        dist.ReduceOp.MAX,
        dist.ReduceOp.PRODUCT,
    ]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        torch.set_default_device(cls.device)

    def _rank_value(self):
        return self.rank + 1

    def _ops(self):
        """Reduce ops valid at this world size.

        PRODUCT over rank+1 is world_size!, and only up to 12! is exactly
        representable in float32.
        """
        for op in self.REDUCE_OPS:
            if op == dist.ReduceOp.PRODUCT and self.num_ranks > 12:
                continue
            yield op

    def _expected_reduce_result(self, op):
        total = sum(range(1, self.num_ranks + 1))
        if op == dist.ReduceOp.SUM:
            return total
        if op == dist.ReduceOp.AVG:
            return total / self.num_ranks
        if op == dist.ReduceOp.MIN:
            return 1
        if op == dist.ReduceOp.MAX:
            return self.num_ranks
        if op == dist.ReduceOp.PRODUCT:
            product = 1
            for i in range(1, self.num_ranks + 1):
                product *= i
            return product
        raise ValueError(f"Unsupported op: {op}")

    def test_allreduce(self):
        for op in self._ops():
            with self.subTest(op=get_op_name(op)):
                tensor = torch.tensor([self._rank_value()], dtype=torch.float32)
                dist.all_reduce(tensor, op=op)
                self.assertEqual(tensor.item(), self._expected_reduce_result(op))

    def test_all_gather(self):
        input_tensor = torch.tensor([self._rank_value()], dtype=torch.float32)
        gather_list = [torch.empty_like(input_tensor) for _ in range(self.num_ranks)]
        dist.all_gather(gather_list, input_tensor)
        self.assertEqual(
            [t.item() for t in gather_list], list(range(1, self.num_ranks + 1))
        )

    def test_all_gather_into_tensor(self):
        input_tensor = torch.tensor([self._rank_value()], dtype=torch.float32)
        output_tensor = torch.empty(self.num_ranks, dtype=torch.float32)
        dist.all_gather_into_tensor(output_tensor, input_tensor)
        self.assertEqual(
            [t.item() for t in output_tensor], list(range(1, self.num_ranks + 1))
        )

    def test_broadcast(self):
        tensor = torch.tensor([self._rank_value()], dtype=torch.float32)
        dist.broadcast(tensor, src=0)
        self.assertEqual(tensor.item(), 1)

    def test_gather(self):
        tensor = torch.tensor([self._rank_value()], dtype=torch.float32)
        gather_list = (
            [torch.empty_like(tensor) for _ in range(self.num_ranks)]
            if self.rank == 0
            else None
        )
        dist.gather(tensor, gather_list=gather_list, dst=0)
        if self.rank == 0:
            self.assertEqual(
                [t.item() for t in gather_list], list(range(1, self.num_ranks + 1))
            )

    def test_scatter(self):
        scatter_list = (
            [torch.tensor([i], dtype=torch.float32) for i in range(self.num_ranks)]
            if self.rank == 0
            else None
        )
        tensor = torch.empty(1, dtype=torch.float32)
        dist.scatter(tensor, scatter_list=scatter_list, src=0)
        self.assertEqual(tensor.item(), self.rank)

    def test_reduce(self):
        for op in self._ops():
            with self.subTest(op=get_op_name(op)):
                tensor = torch.tensor([self._rank_value()], dtype=torch.float32)
                dist.reduce(tensor, dst=0, op=op)
                if self.rank == 0:
                    self.assertEqual(
                        tensor.item(), self._expected_reduce_result(op)
                    )

    def test_reduce_scatter(self):
        for op in self._ops():
            with self.subTest(op=get_op_name(op)):
                input_tensor = [
                    torch.tensor([self._rank_value()], dtype=torch.float32)
                    for _ in range(self.num_ranks)
                ]
                output_tensor = torch.empty(1, dtype=torch.float32)
                dist.reduce_scatter(output_tensor, input_tensor, op=op)
                self.assertEqual(
                    output_tensor.item(), self._expected_reduce_result(op)
                )

    def test_reduce_scatter_tensor(self):
        for op in self._ops():
            with self.subTest(op=get_op_name(op)):
                input_tensor = torch.full(
                    (self.num_ranks,), self._rank_value(), dtype=torch.float32
                )
                output_tensor = torch.empty(1, dtype=torch.float32)
                dist.reduce_scatter_tensor(output_tensor, input_tensor, op=op)
                self.assertEqual(
                    output_tensor.item(), self._expected_reduce_result(op)
                )

    def test_all_to_all(self):
        input_tensor = [
            torch.tensor([self._rank_value()], dtype=torch.float32)
            for _ in range(self.num_ranks)
        ]
        output_tensor = [
            torch.empty(1, dtype=torch.float32) for _ in range(self.num_ranks)
        ]
        dist.all_to_all(output_tensor, input_tensor)
        self.assertEqual(
            [t.item() for t in output_tensor], list(range(1, self.num_ranks + 1))
        )

    def test_all_to_all_single(self):
        input_tensor = torch.full(
            (self.num_ranks,), self._rank_value(), dtype=torch.float32
        )
        output_tensor = torch.empty([self.num_ranks], dtype=torch.float32)
        dist.all_to_all_single(output_tensor, input_tensor)
        self.assertEqual(
            [t.item() for t in output_tensor], list(range(1, self.num_ranks + 1))
        )

    def test_all_to_all_single_with_split_sizes(self):
        """Rank r sends r+1 elements to every peer, so receives i+1 from i."""
        input_split_sizes = [self.rank + 1] * self.num_ranks
        output_split_sizes = [i + 1 for i in range(self.num_ranks)]

        input_tensor = torch.empty(sum(input_split_sizes), dtype=torch.float32)
        offset = 0
        for dst in range(self.num_ranks):
            input_tensor[offset : offset + input_split_sizes[dst]].fill_(
                self.rank + dst
            )
            offset += input_split_sizes[dst]

        output_tensor = torch.empty(sum(output_split_sizes), dtype=torch.float32)
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )

        offset = 0
        for src in range(self.num_ranks):
            section = output_tensor[offset : offset + output_split_sizes[src]]
            expected = torch.full_like(section, src + self.rank)
            self.assertTrue(
                torch.equal(section, expected),
                f"Mismatch in section from rank {src}: "
                f"got {section}, expected {expected}",
            )
            offset += output_split_sizes[src]

    def test_send_recv(self):
        send_rank = (self.rank + 1) % self.num_ranks
        recv_rank = (self.rank + self.num_ranks - 1) % self.num_ranks
        send_tensor = torch.tensor([self.rank], dtype=torch.float32)
        recv_tensor = torch.empty(1, dtype=torch.float32)
        if self.rank % 2 == 0:
            dist.send(send_tensor, dst=send_rank)
            dist.recv(recv_tensor, src=recv_rank)
        else:
            dist.recv(recv_tensor, src=recv_rank)
            dist.send(send_tensor, dst=send_rank)
        self.assertEqual(recv_tensor.item(), recv_rank)

    def test_barrier(self):
        dist.barrier()


if __name__ == "__main__":
    run_tests()
