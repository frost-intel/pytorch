# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms FSDPCommTest.py onto xccl2.
#
# Mesh construction moves from torchcomms.device_mesh to the stock one; the
# sharded-vs-reference comparison is unchanged.
#
# set_gradient_divide_factor(1.0) is kept from the original: it pins the
# gradient reduction to a plain SUM. FSDP's default pre/post-divide would need
# ReduceOp.AVG on the reduce_scatter, which oneCCL lowers to SUM (oneCCL#195).

import copy

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard

from common import DEVICE_TYPE, run_tests, Xccl2TestBase


class FSDPCommTest(Xccl2TestBase):
    def test_training(self) -> None:
        device_mesh = init_device_mesh(
            DEVICE_TYPE, (self.num_ranks,), mesh_dim_names=("main",)
        )

        torch.manual_seed(42)
        dim0 = 4
        nlayer = 2
        model = nn.Sequential(
            *[
                nn.Linear(dim0, dim0, bias=False, device=self.device)
                for _ in range(nlayer)
            ]
        )
        ref_model = copy.deepcopy(model)
        for layer in model:
            fully_shard(layer, mesh=device_mesh)
            if isinstance(layer, FSDPModule):
                layer.set_gradient_divide_factor(1.0)
        fully_shard(model, mesh=device_mesh)

        optim = torch.optim.Adam(model.parameters(), lr=0.05)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=0.05)
        inp = torch.randn((4, dim0), device=self.device)

        for _ in range(10):
            loss = model(inp).sum()
            ref_loss = ref_model(inp).sum()
            self.assertTrue(torch.allclose(loss, ref_loss, atol=1e-7, rtol=1e-5))

            loss.backward()
            ref_loss.backward()
            optim.step()
            ref_optim.step()
            optim.zero_grad()
            ref_optim.zero_grad()


if __name__ == "__main__":
    run_tests()
