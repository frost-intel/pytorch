#!/bin/bash
# Run the xccl2 integration tests.
#
# Port of torchcomms' scripts/run_tests_integration_xccl_py.sh. The original
# drives the torchcomms XCCL backend; this drives the in-tree xccl2 c10d
# backend over the same collectives.
#
#   NPROC_PER_NODE=4 bash run_tests_integration_xccl2_py.sh
#
# TEST_FULL_SWEEP=0 trims the count/dtype/op sweeps for a quick smoke run.

set -ex

cd "$(dirname "$0")"

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export TEST_BACKEND=${TEST_BACKEND:-xccl2}
export TEST_DEVICE=${TEST_DEVICE:-xpu}
export TEST_FULL_SWEEP=${TEST_FULL_SWEEP:-1}

tests=(
    AllGatherSingleTest.py
    AllGatherTest.py
    AllGatherVTest.py
    AllReduceTest.py
    AllToAllSingleTest.py
    AllToAllTest.py
    AllToAllvSingleTest.py
    BackendWrapperAllGatherAliasedTest.py
    BackendWrapperCoalescingTest.py
    BackendWrapperShutdownTest.py
    BarrierTest.py
    BatchSendRecvTest.py
    BroadcastTest.py
    C10dBatchIsendIrecvTest.py
    C10dTorchCommTest.py
    DDPCommTest.py
    DeviceMeshTest.py
    DPTPCommTest.py
    FinalizeWarningTest.py
    FSDPCommTest.py
    FullgraphCompileAutogradTest.py
    FullgraphCompileTest.py
    GatherTest.py
    MemPoolTest.py
    MultiCommTest.py
    ObjColTest.py
    OptionsTest.py
    ReduceScatterSingleTest.py
    ReduceScatterTest.py
    ReduceScatterVTest.py
    ReduceTest.py
    SendRecvTest.py
    ScatterTest.py
    SplitTest.py
    TPCommTest.py
    WaitBlockingTest.py
)

for test_file in "${tests[@]}"; do
    torchrun --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" -m pytest -v -s "$test_file"
done
