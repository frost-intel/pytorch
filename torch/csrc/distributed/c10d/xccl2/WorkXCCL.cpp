// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_XCCL

#include <torch/csrc/distributed/c10d/xccl2/WorkXCCL.hpp>

#include <ATen/core/ivalue.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/xccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/xccl2/ProcessGroupXCCL.hpp>
#include <torch/csrc/distributed/c10d/xccl2/TracingGuard.hpp>

namespace c10d::xccl2 {

std::unique_ptr<at::xpu::XPUEvent> XcclEventPool::get() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!pool_.empty()) {
    std::unique_ptr<at::xpu::XPUEvent> event = std::move(pool_.front());
    pool_.pop();
    return event;
  }

  // Create new event if pool is empty
  return std::make_unique<at::xpu::XPUEvent>();
}

void XcclEventPool::put(std::unique_ptr<at::xpu::XPUEvent> event) {
  if (!event) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);

  if (pool_.size() < max_size_) {
    pool_.push(std::move(event));
  }
  // Otherwise the pool is full and the event is destroyed with the unique_ptr.
}

void XcclEventPool::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  while (!pool_.empty()) {
    pool_.pop();
  }
}

namespace {

XcclCommInfo makeCommInfo(ProcessGroupXCCL* comm) {
  return XcclCommInfo{
      std::string(comm->getCommName()),
      comm->getRank(),
      comm->getSize(),
      comm->getDevice()};
}

} // namespace

WorkXCCL::WorkXCCL(
    ProcessGroupXCCL* comm,
    c10::xpu::XPUStream stream,
    std::chrono::milliseconds timeout_ms,
    const std::vector<at::Tensor>& inputTensors)
    : inputTensors_(inputTensors),
      comm_info_(makeCommInfo(comm)),
      event_pool_(comm->getEventPool()),
      stream_(stream),
      timeout_ms_(timeout_ms) {
  start_event_ = event_pool_->get();
  end_event_ = event_pool_->get();
  // Events are recorded around the actual XCCL operations by the backend.
}

WorkXCCL::WorkXCCL(
    ProcessGroupXCCL* comm,
    c10::xpu::XPUStream stream,
    std::chrono::milliseconds timeout_ms,
    at::Tensor inputTensor)
    : inputTensor_(std::move(inputTensor)),
      comm_info_(makeCommInfo(comm)),
      event_pool_(comm->getEventPool()),
      stream_(stream),
      timeout_ms_(timeout_ms) {
  start_event_ = event_pool_->get();
  end_event_ = event_pool_->get();
}

WorkXCCL::~WorkXCCL() {
  event_pool_->put(std::move(start_event_));
  event_pool_->put(std::move(end_event_));
}

void WorkXCCL::recordFunctionStart(std::string_view coll_name) {
  recordFunction_.emplace(at::RecordScope::USER_SCOPE);
  if (!recordFunction_->isActive()) {
    return;
  }

  // Passing the input tensors to recordFunction allows for shape information in
  // the profiling output.
  if (!inputTensors_.empty()) {
    std::vector<c10::IValue> inputs;
    inputs.reserve(inputTensors_.size());
    for (const auto& tensor : inputTensors_) {
      inputs.emplace_back(tensor);
    }
    recordFunction_->before(
        coll_name,
        c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
  } else if (inputTensor_.defined()) {
    recordFunction_->before(
        coll_name, c10::ArrayRef<const c10::IValue>(inputTensor_));
  } else {
    recordFunction_->before(coll_name, c10::ArrayRef<const c10::IValue>{});
  }
}

void WorkXCCL::recordStart(std::string_view coll_name) {
  recordFunctionStart(coll_name);
  start_event_->record(stream_);
}

void WorkXCCL::recordEnd() {
  end_event_->record(stream_);

  if (recordFunction_ && recordFunction_->isActive()) {
    recordFunction_->end();
  }
}

WorkXCCL::WorkStatus WorkXCCL::checkStatus() {
  if (status() == WorkStatus::COMPLETED || status() == WorkStatus::ERROR ||
      status() == WorkStatus::TIMEDOUT) {
    return status();
  }

  // Step 1: if the start event has not been observed to complete yet, query it
  // and remember when it did. The timeout is measured from that point, not from
  // when the work was created, so time spent queued behind other collectives is
  // not charged against this work's timeout.
  if (!start_completed_time_.has_value()) {
    try {
      if (start_event_->query()) {
        start_completed_time_ = std::chrono::steady_clock::now();
        setStatus(WorkStatus::INPROGRESS);
      }
    } catch (const std::exception& e) {
      TC_LOG(ERROR, &comm_info_) << "XPU error during start event query: "
                                << e.what();
      setStatus(WorkStatus::ERROR);
    }
  }
  if (status() == WorkStatus::NOT_STARTED || status() == WorkStatus::ERROR) {
    return status();
  }

  // Step 2: the collective has started, so poll the end event for completion.
  bool end_completed = false;
  try {
    end_completed = end_event_->query();
  } catch (const std::exception& e) {
    TC_LOG(ERROR, &comm_info_) << "XPU error during end event query: "
                               << e.what();
    setStatus(WorkStatus::ERROR);
    return status();
  }

  if (end_completed) {
    setStatus(WorkStatus::COMPLETED);
  } else {
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_completed_time_.value());

    if (elapsed_milliseconds > timeout_ms_) {
      TC_LOG(ERROR, &comm_info_) << "Operation timed out after "
                                 << elapsed_milliseconds.count() << " ms";
      setStatus(WorkStatus::TIMEDOUT);
    }
  }
  return status();
}

bool WorkXCCL::isCompleted() {
  return checkStatus() == WorkStatus::COMPLETED;
}

bool WorkXCCL::isSuccess() const {
  WorkStatus s = status();
  return s != WorkStatus::ERROR && s != WorkStatus::TIMEDOUT;
}

void WorkXCCL::synchronizeInternal() {
  WorkStatus local_state = status();
  if (local_state == WorkStatus::COMPLETED ||
      local_state == WorkStatus::ERROR || local_state == WorkStatus::TIMEDOUT) {
    return;
  }

  TracingGuard tracingGuard(
      comm_info_.name, comm_info_.size, "wait", comm_info_.rank);

  // Make the current stream wait for the end event recorded on the work's
  // stream, ordering subsequent current-stream ops after this collective.
  auto current_stream =
      c10::xpu::getCurrentXPUStream(comm_info_.device.index());
  end_event_->block(current_stream);

  // For a synchronous barrier, host-block the CPU thread until prior
  // current-stream work has completed rather than merely stream-ordering it:
  // callers use barrier() to flush async device work before proceeding, and a
  // stream-order-only barrier lets the next collective race that work.
  if (hostBlocking_) {
    current_stream.synchronize();
  }

  // Release tensor references. The XPU caching allocator manages stream
  // semantics and will not reclaim memory until the stream operations complete.
  inputTensors_.clear();
  inputTensor_.reset();
}

bool WorkXCCL::wait(std::chrono::milliseconds /*timeout*/) {
  // Unlike c10d's default wait(), this does not block the CPU: for XPU work it
  // is sufficient (and matches upstream torchcomms) to order the current stream
  // after the collective. The timeout arg is honored by the watchdog, not here.
  synchronize();
  return true;
}

void WorkXCCL::synchronize() {
  synchronizeInternal();
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<WorkXCCL>::unsafe_reclaim_from_nonowning(this));
  }
}

std::vector<at::Tensor> WorkXCCL::result() {
  return outputs_;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkXCCL::getFuture() {
  if (future_) {
    return future_;
  }

  std::vector<c10::Device> devices;
  for (const auto& tensor : outputs_) {
    if (tensor.device().type() != c10::DeviceType::CPU) {
      devices.push_back(tensor.device());
      break;
    }
  }
  future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);

  // Order the current stream after the collective before completing the future
  // so consumers observing the future see correct results.
  synchronize();

  if (!outputs_.empty() && !devices.empty()) {
    c10::OptionalDeviceGuard guard(outputs_[0].device());
    future_->markCompleted(c10::IValue(outputs_));
  } else {
    future_->markCompleted(c10::IValue(outputs_));
  }
  return future_;
}

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL
