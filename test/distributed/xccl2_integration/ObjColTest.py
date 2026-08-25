# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms ObjColTest.py onto the xccl2 c10d backend.
# torchcomms' objcol.* helpers map to the object collectives that c10d already
# ships; they serialise to byte tensors and ride the same backend collectives.

import torch.distributed as dist

from common import run_tests, Xccl2TestBase

ROOT_RANK = 0


def make_object(rank):
    return {"rank": rank, "payload": [rank] * 4, "tag": f"rank-{rank}"}


class ObjColTest(Xccl2TestBase):
    def test_all_gather_object(self):
        gathered = [None] * self.num_ranks
        dist.all_gather_object(gathered, make_object(self.rank))
        self.assertEqual(gathered, [make_object(r) for r in range(self.num_ranks)])

    def test_gather_object(self):
        output = [None] * self.num_ranks if self.rank == ROOT_RANK else None
        dist.gather_object(make_object(self.rank), output, dst=ROOT_RANK)
        if self.rank == ROOT_RANK:
            self.assertEqual(output, [make_object(r) for r in range(self.num_ranks)])

    def test_broadcast_object_list(self):
        objects = (
            [make_object(ROOT_RANK), "payload"]
            if self.rank == ROOT_RANK
            else [None, None]
        )
        dist.broadcast_object_list(objects, src=ROOT_RANK, device=self.device)
        self.assertEqual(objects, [make_object(ROOT_RANK), "payload"])

    def test_scatter_object_list(self):
        input_list = (
            [make_object(r) for r in range(self.num_ranks)]
            if self.rank == ROOT_RANK
            else None
        )
        output_list = [None]
        dist.scatter_object_list(output_list, input_list, src=ROOT_RANK)
        self.assertEqual(output_list[0], make_object(self.rank))

    def test_send_recv_object_list(self):
        send_rank = (self.rank + 1) % self.num_ranks
        recv_rank = (self.rank - 1) % self.num_ranks

        if self.rank % 2 == 0:
            dist.send_object_list([make_object(self.rank)], dst=send_rank)
            received = [None]
            dist.recv_object_list(received, src=recv_rank)
        else:
            received = [None]
            dist.recv_object_list(received, src=recv_rank)
            dist.send_object_list([make_object(self.rank)], dst=send_rank)

        self.assertEqual(received[0], make_object(recv_rank))


if __name__ == "__main__":
    run_tests()
