# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms DPTPCommTest.py onto xccl2.
#
# The original carves the DP and TP sub-communicators out by hand with
# comm.split(dp_ranks) / comm.split(tp_ranks) and hands them to the torchcomms
# mesh. The stock init_device_mesh builds the 2D mesh and its sub-groups
# itself, so the split calls disappear. xccl2 has no PG-level split yet, so the
# mesh falls back to new_group, which creates fresh communicators -- correct,
# just not the fast path.

import copy
import unittest
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable import checkpoint
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
    SequenceParallel,
)

from common import DEVICE_TYPE, run_tests, Xccl2TestBase


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        device: Optional[torch.device] = None,
        *,
        bias: bool = True,
        dim_multiplier: int = 4,
    ):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim_multiplier * dim, device=device, bias=bias)
        self.out_proj = nn.Linear(dim_multiplier * dim, dim, device=device, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z = F.relu(z)
        z = self.out_proj(z)
        z = F.relu(z)
        return z


class MLPStack(nn.Sequential):
    def __init__(self, mlp_dim: int, *, with_seq_parallel: bool = False):
        modules: list[nn.Module] = [
            MLP(mlp_dim, dim_multiplier=4),
            MLP(mlp_dim),
            MLP(mlp_dim, dim_multiplier=4),
        ]
        if with_seq_parallel:
            modules.append(nn.LayerNorm(mlp_dim, bias=False))
        super().__init__(*modules)
        self.with_seq_parallel = with_seq_parallel

    def parallelize(
        self,
        tp_mesh: DeviceMesh,
        dp_mesh: DeviceMesh,
        use_activation_checkpointing: bool,
        **fsdp_kwargs,
    ) -> "MLPStack":
        parallelize_plan = {
            # use_local_output=False keeps DTensors so uneven activation dims
            # survive the layer boundary.
            "0.in_proj": ColwiseParallel(use_local_output=False),
            "0.out_proj": RowwiseParallel(use_local_output=False),
            "1.in_proj": ColwiseParallel(use_local_output=False),
            "1.out_proj": RowwiseParallel(use_local_output=False),
            "2.in_proj": ColwiseParallel(use_local_output=False),
            "2.out_proj": RowwiseParallel(output_layouts=Shard(1))
            if self.with_seq_parallel
            else RowwiseParallel(),
        }
        if self.with_seq_parallel:
            parallelize_plan["3"] = SequenceParallel(sequence_dim=1)
        parallelize_module(
            self, device_mesh=tp_mesh, parallelize_plan=parallelize_plan
        )
        for module in self:
            if isinstance(module, nn.LayerNorm):
                continue
            if use_activation_checkpointing:
                checkpoint(module)
            fully_shard(module, mesh=dp_mesh, **fsdp_kwargs)
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)
        return self


class DPTPCommTest(Xccl2TestBase):
    @unittest.skipIf(
        torch.accelerator.device_count() < 4, "2D parallel needs 4+ devices"
    )
    def test_training(self) -> None:
        dp_degree = 2
        tp_degree = self.num_ranks // dp_degree
        device_mesh_2d = init_device_mesh(
            DEVICE_TYPE, (dp_degree, tp_degree), mesh_dim_names=("dp", "tp")
        )
        dp_pg = device_mesh_2d.get_group("dp")

        mlp_dim = 16
        lr = 1e-4
        torch.manual_seed(42)
        model = MLPStack(mlp_dim).to(self.device)

        ref_model = copy.deepcopy(model).to(self.device)
        model.parallelize(
            device_mesh_2d["tp"],
            device_mesh_2d["dp"],
            False,
            reshard_after_forward=False,
        )
        # The reference still needs data parallelism to sync its gradients.
        for layer in ref_model:
            fully_shard(layer, mesh=device_mesh_2d["dp"])
        fully_shard(ref_model, mesh=device_mesh_2d["dp"])

        optim = torch.optim.Adam(model.parameters(), lr=lr, foreach=False)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=lr, foreach=False)

        torch.manual_seed(42 + dp_pg.rank() + 1)
        for iter_idx in range(10):
            inp = torch.randn((8, mlp_dim), device=self.device)
            losses: list[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertTrue(
                torch.allclose(losses[0], losses[1], atol=1e-7, rtol=1e-5)
            )

        self.sync_device()


if __name__ == "__main__":
    run_tests()
