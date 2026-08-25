# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms DDPCommTest.py onto xccl2.
#
# The original builds a mesh from a torchcomms comm object
# (torchcomms.device_mesh.init_device_mesh(mesh_dim_comms=(comm,), ...)) and
# pulls a process group back out of it. xccl2 is a c10d backend, so the stock
# torch.distributed.device_mesh builds the mesh over the default group instead
# and the DDP body is unchanged.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.parallel import DistributedDataParallel as DDP

from common import DEVICE_TYPE, run_tests, Xccl2TestBase


class DDPCommTest(Xccl2TestBase):
    def test_training(self) -> None:
        device_mesh = init_device_mesh(
            DEVICE_TYPE, (self.num_ranks,), mesh_dim_names=("main",)
        )
        pg = device_mesh.get_group("main")

        torch.manual_seed(42)
        dim0 = 4
        nlayer = 2
        model = nn.Sequential(
            *[
                nn.Linear(dim0, dim0, bias=False, device=self.device)
                for _ in range(nlayer)
            ]
        )
        model = DDP(model, process_group=pg)

        optim = torch.optim.Adam(model.parameters(), lr=0.05)
        inp = torch.randn((4, dim0)).to(self.device)

        prev_loss = None
        for i in range(10):
            loss = model(inp).sum()
            loss.backward()
            optim.step()
            optim.zero_grad()

            loss_val = loss.item()
            if prev_loss is not None and i < 5:
                self.assertNotEqual(
                    loss_val, prev_loss, f"loss did not change at step {i}"
                )
            prev_loss = loss_val

        self.sync_device()


if __name__ == "__main__":
    run_tests()
