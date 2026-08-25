# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms DeviceMeshTest.py onto xccl2.
#
# The original builds meshes out of torchcomms comm objects and their splits.
# xccl2 reaches DeviceMesh through the stock c10d path, so these check the same
# properties -- mesh shape, per-dim groups, collectives scoped to a mesh dim,
# and flattening -- built the in-tree way.
#
# The original's test_backend_wrapper_split_group has no counterpart: it drives
# BackendWrapper.split_group, and xccl2 has no PG-level split yet (see
# SplitTest.py for that gap).

import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from common import DEVICE_TYPE, run_tests, Xccl2TestBase


class DeviceMeshTest(Xccl2TestBase):
    def test_init(self) -> None:
        mesh = init_device_mesh(
            DEVICE_TYPE, (self.num_ranks,), mesh_dim_names=("main",)
        )
        self.assertEqual(mesh.ndim, 1)
        self.assertEqual(mesh.size(), self.num_ranks)
        self.assertEqual(mesh.get_group("main").size(), self.num_ranks)

    def test_all_reduce_over_mesh_group(self) -> None:
        mesh = init_device_mesh(
            DEVICE_TYPE, (self.num_ranks,), mesh_dim_names=("main",)
        )
        pg = mesh.get_group("main")
        tensor = torch.ones(16, device=self.device) * (self.rank + 1)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=pg)
        expected = self.num_ranks * (self.num_ranks + 1) / 2
        torch.testing.assert_close(
            tensor.cpu(), torch.full_like(tensor.cpu(), expected)
        )

    @unittest.skipIf(
        torch.accelerator.device_count() < 4, "2D mesh needs 4+ devices"
    )
    def test_2_d_parallel(self) -> None:
        dp_degree = 2
        tp_degree = self.num_ranks // dp_degree
        mesh = init_device_mesh(
            DEVICE_TYPE, (dp_degree, tp_degree), mesh_dim_names=("dp", "tp")
        )
        self.assertEqual(mesh.ndim, 2)
        self.assertEqual(mesh["dp"].size(), dp_degree)
        self.assertEqual(mesh["tp"].size(), tp_degree)

        # A collective on the tp dim must only reduce within that row.
        tp_pg = mesh.get_group("tp")
        tensor = torch.ones(16, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=tp_pg)
        torch.testing.assert_close(
            tensor.cpu(), torch.full_like(tensor.cpu(), float(tp_degree))
        )

        dp_pg = mesh.get_group("dp")
        tensor = torch.ones(16, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=dp_pg)
        torch.testing.assert_close(
            tensor.cpu(), torch.full_like(tensor.cpu(), float(dp_degree))
        )

    @unittest.skipIf(
        torch.accelerator.device_count() < 8, "3D mesh needs 8+ devices"
    )
    def test_n_d_parallel(self) -> None:
        mesh = init_device_mesh(
            DEVICE_TYPE,
            (2, 2, self.num_ranks // 4),
            mesh_dim_names=("dp", "cp", "tp"),
        )
        self.assertEqual(mesh.ndim, 3)
        for name, expected in (
            ("dp", 2),
            ("cp", 2),
            ("tp", self.num_ranks // 4),
        ):
            self.assertEqual(mesh[name].size(), expected)
            tensor = torch.ones(8, device=self.device)
            dist.all_reduce(tensor, group=mesh.get_group(name))
            torch.testing.assert_close(
                tensor.cpu(), torch.full_like(tensor.cpu(), float(expected))
            )

    @unittest.skipIf(
        torch.accelerator.device_count() < 4, "flatten needs a 2D mesh"
    )
    def test_flatten(self) -> None:
        dp_degree = 2
        tp_degree = self.num_ranks // dp_degree
        mesh = init_device_mesh(
            DEVICE_TYPE, (dp_degree, tp_degree), mesh_dim_names=("dp", "tp")
        )
        flat = mesh._flatten("world")
        self.assertEqual(flat.size(), self.num_ranks)

        tensor = torch.ones(16, device=self.device)
        dist.all_reduce(tensor, group=flat.get_group())
        torch.testing.assert_close(
            tensor.cpu(), torch.full_like(tensor.cpu(), float(self.num_ranks))
        )


if __name__ == "__main__":
    run_tests()
