# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms OptionsTest.py onto xccl2.
#
# The original checks that every TorchComm collective accepts its per-call
# options -- a `hints` map and a `timeout` -- as Python keyword arguments. It
# verifies acceptance only, not plumbing; the original carries a TODO saying so.
#
# c10d has no per-call hints. Timeouts and hints are group-level on xccl2:
# ProcessGroupXCCL::Options carries is_high_priority_stream and a hints map,
# bound as ProcessGroupXCCL2.Options. So the equivalent check is that the
# options object is constructible, accepted by init, and that the group still
# runs every collective afterwards.

import datetime

import torch
import torch.distributed as dist

from common import BACKEND, DEVICE_TYPE, get_rank_and_size, local_rank, make_store, run_tests, Xccl2TestBase

TENSOR_COUNT = 4


class OptionsTest(Xccl2TestBase):
    @classmethod
    def setUpClass(cls) -> None:
        # This file owns its group: the point is to pass Options at init.
        cls.rank, cls.num_ranks = get_rank_and_size()
        index = local_rank()
        if DEVICE_TYPE == "xpu":
            torch.xpu.set_device(index)
        cls.device = torch.device(DEVICE_TYPE, index)

    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def setUp(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _init_with_options(self, name, **option_kwargs):
        opts = dist.ProcessGroupXCCL2.Options()
        for key, value in option_kwargs.items():
            setattr(opts, key, value)
        dist.init_process_group(
            backend=BACKEND,
            store=make_store(name),
            rank=self.rank,
            world_size=self.num_ranks,
            pg_options=opts,
            timeout=datetime.timedelta(seconds=600),
        )

    def test_options_type_is_bound(self):
        """ProcessGroupXCCL2.Options is exposed and has xccl2's own fields."""
        opts = dist.ProcessGroupXCCL2.Options()
        self.assertTrue(hasattr(opts, "is_high_priority_stream"))
        self.assertTrue(hasattr(opts, "hints"))

    def test_high_priority_stream(self):
        self._init_with_options(
            "options_high_priority", is_high_priority_stream=True
        )
        tensor = torch.ones(TENSOR_COUNT, device=self.device) * float(self.rank + 1)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = self.num_ranks * (self.num_ranks + 1) / 2
        torch.testing.assert_close(
            tensor.cpu(), torch.full_like(tensor.cpu(), expected)
        )

    def test_hints_map_accepted(self):
        self._init_with_options(
            "options_hints", hints={"is_high_priority_stream": "1"}
        )
        tensor = torch.ones(TENSOR_COUNT, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(
            tensor.cpu(), torch.full_like(tensor.cpu(), float(self.num_ranks))
        )

    def test_all_collectives_under_options(self):
        """Every collective the original sweeps still runs with Options set."""
        self._init_with_options("options_all_collectives")
        options = {"dtype": torch.float, "device": self.device}
        n = self.num_ranks

        send_tensor = torch.ones(TENSOR_COUNT, **options) * float(self.rank + 1)
        dist.all_reduce(send_tensor, op=dist.ReduceOp.SUM)

        dist.broadcast(torch.ones(TENSOR_COUNT, **options), src=0)

        gather_out = [torch.zeros(TENSOR_COUNT, **options) for _ in range(n)]
        dist.all_gather(gather_out, torch.ones(TENSOR_COUNT, **options))

        dist.reduce_scatter(
            torch.zeros(TENSOR_COUNT, **options),
            [torch.ones(TENSOR_COUNT, **options) for _ in range(n)],
            op=dist.ReduceOp.SUM,
        )

        dist.all_to_all_single(
            torch.zeros(TENSOR_COUNT * n, **options),
            torch.ones(TENSOR_COUNT * n, **options),
        )

        dist.all_to_all_single(
            torch.zeros(TENSOR_COUNT * n, **options),
            torch.ones(TENSOR_COUNT * n, **options),
            output_split_sizes=[TENSOR_COUNT] * n,
            input_split_sizes=[TENSOR_COUNT] * n,
        )

        dist.reduce(torch.ones(TENSOR_COUNT, **options), dst=0, op=dist.ReduceOp.SUM)

        if self.num_ranks >= 2:
            send_rank = (self.rank + 1) % n
            recv_rank = (self.rank + n - 1) % n
            payload = torch.ones(TENSOR_COUNT, **options) * float(self.rank + 1)
            received = torch.zeros(TENSOR_COUNT, **options)
            if self.rank % 2 == 0:
                dist.send(payload, dst=send_rank)
                dist.recv(received, src=recv_rank)
            else:
                dist.recv(received, src=recv_rank)
                dist.send(payload, dst=send_rank)

        dist.barrier()


if __name__ == "__main__":
    run_tests()
