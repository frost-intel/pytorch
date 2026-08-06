# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms TPCommTest.py onto xccl2.
#
# Mesh construction moves from torchcomms.device_mesh to the stock one; the
# colwise/rowwise parallelised MLP and its comparison against the unsharded
# reference are unchanged.

import copy
import unittest

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)

from common import DEVICE_TYPE, run_tests, Xccl2TestBase


class MLPModule(nn.Module):
    def __init__(self, device, bias: bool = True):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = nn.Linear(10, 16, bias=bias, device=device)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 10, bias=bias, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

    def reset_parameters(self):
        self.net1.reset_parameters()
        self.net2.reset_parameters()


class TPCommTest(Xccl2TestBase):
    def _compare_params(self, local_module, dist_module, compare_grad=False):
        replicate = [Replicate()]
        for name, param in local_module.named_parameters():
            dist_param = dist_module.get_parameter(name)
            param = param.grad if compare_grad else param
            dist_param = dist_param.grad if compare_grad else dist_param
            tp_param_full_tensor = dist_param.redistribute(
                device_mesh=dist_param.device_mesh, placements=replicate
            ).to_local()
            self.assertTrue(
                torch.equal(param, tp_param_full_tensor),
                f"parameter {name} diverged",
            )

    @unittest.skipIf(
        torch.accelerator.device_count() < 2, "tensor parallel needs 2+ devices"
    )
    def test_training(self) -> None:
        device_mesh = init_device_mesh(DEVICE_TYPE, (self.num_ranks,))

        inp_size = [12, 10]
        model = MLPModule(self.device)
        torch.manual_seed(0)

        model_tp = copy.deepcopy(model)
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {"net1": ColwiseParallel(), "net2": RowwiseParallel()},
        )
        lr = 8e-4
        local_optim = torch.optim.SGD(model.parameters(), lr=lr)
        dist_optim = torch.optim.SGD(model_tp.parameters(), lr=lr)
        self._compare_params(model, model_tp)

        for _ in range(30):
            inp = torch.rand(*inp_size, device=self.device)
            output = model(inp)
            tp_output = model_tp(inp)
            tp_output = (
                tp_output.redistribute(
                    tp_output.device_mesh, [Replicate()]
                ).to_local()
                if isinstance(tp_output, DTensor)
                else tp_output
            )
            loss_diff = output.sum() - tp_output.sum()
            self.assertLess(abs(loss_diff.item()), 1e-5)

            output.sum().backward()
            tp_output.sum().backward()
            self._compare_params(model, model_tp, compare_grad=True)

            local_optim.step()
            dist_optim.step()
            self._compare_params(model, model_tp)

            local_optim.zero_grad()
            dist_optim.zero_grad()


if __name__ == "__main__":
    run_tests()
