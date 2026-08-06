# Owner(s): ["oncall: distributed"]
#
# Port of torchcomms FullgraphCompileAutogradTest.py onto xccl2.
#
# The original tests torchcomms' own async-collective machinery:
# torch.ops.torchcomms.torchcomm_wait_tensors (and its in-place form) and the
# TorchCommsAsyncTensor subclass, checking that autograd flows through a
# pending collective and that the wait is transparent to the graph.
#
# xccl2 registers no such ops. c10d's equivalents are
# torch.ops._c10d_functional.wait_tensor and AsyncCollectiveTensor, so the
# ported tests target those: same property (gradients survive an async
# collective and its wait), in-tree op set.

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_reduce as functional_all_reduce,
    AsyncCollectiveTensor,
)

from common import run_tests, Xccl2TestBase


class FullgraphCompileAutogradTest(Xccl2TestBase):
    def setUp(self):
        torch._dynamo.reset()

    def _rank_tensor(self, count=8, requires_grad=True):
        return torch.ones(
            count, dtype=torch.float, device=self.device, requires_grad=requires_grad
        ) * (self.rank + 1)

    def test_wait_tensor_backward(self):
        """wait_tensor must be transparent to autograd."""
        x = torch.ones(8, device=self.device, requires_grad=True)
        y = torch.ops._c10d_functional.wait_tensor(x * 2)
        y.sum().backward()
        torch.testing.assert_close(
            x.grad.cpu(), torch.full_like(x.grad.cpu(), 2.0)
        )

    def test_wait_tensor_grad_chain(self):
        """A wait in the middle of a chain must not sever the gradient."""
        x = torch.ones(8, device=self.device, requires_grad=True)
        y = x * 3
        waited = torch.ops._c10d_functional.wait_tensor(y)
        (waited * 2).sum().backward()
        torch.testing.assert_close(
            x.grad.cpu(), torch.full_like(x.grad.cpu(), 6.0)
        )

    def test_async_collective_tensor_is_produced(self):
        """A functional collective returns an AsyncCollectiveTensor."""
        x = torch.ones(8, device=self.device)
        out = functional_all_reduce(x, "sum", list(range(self.num_ranks)))
        self.assertIsInstance(out, AsyncCollectiveTensor)

    def test_async_collective_tensor_materializes(self):
        """Touching the result waits and yields the reduced value."""
        x = torch.ones(8, device=self.device)
        out = functional_all_reduce(x, "sum", list(range(self.num_ranks)))
        torch.testing.assert_close(
            out.cpu(), torch.full_like(x.cpu(), float(self.num_ranks))
        )

    def test_async_collective_tensor_backward(self):
        """Gradients flow back through a functional all_reduce."""
        x = torch.ones(8, device=self.device, requires_grad=True)
        out = functional_all_reduce(x * 2, "sum", list(range(self.num_ranks)))
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_gradient_accumulation_across_waits(self):
        """Two waited branches off one leaf accumulate into a single grad."""
        x = torch.ones(8, device=self.device, requires_grad=True)
        y1 = torch.ops._c10d_functional.wait_tensor(x.clone())
        y2 = torch.ops._c10d_functional.wait_tensor(x.clone())
        (y1.sum() + y2.sum()).backward()
        torch.testing.assert_close(
            x.grad.cpu(), torch.full_like(x.grad.cpu(), 2.0)
        )

    def test_wait_tensor_requires_grad_propagates(self):
        x = torch.ones(8, device=self.device, requires_grad=True)
        self.assertTrue(torch.ops._c10d_functional.wait_tensor(x).requires_grad)

        z = torch.ones(8, device=self.device, requires_grad=False)
        self.assertFalse(torch.ops._c10d_functional.wait_tensor(z).requires_grad)

    def test_compiled_collective_backward(self):
        """The whole forward+collective compiles and still trains."""
        graphs = []

        def compiler(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        def fn(t):
            out = t * 2
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            return out.sum()

        x = torch.ones(8, device=self.device, requires_grad=True)
        compiled = torch.compile(fn, fullgraph=True, backend=compiler)
        compiled(x).backward()

        self.assertEqual(len(graphs), 1, "did not compile into one graph")
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    run_tests()
