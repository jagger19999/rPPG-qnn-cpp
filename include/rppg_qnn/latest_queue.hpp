#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <utility>

namespace rppg_qnn {

template <typename T>
class LatestQueue {
 public:
  bool push(T value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return false;
      }
      value_.emplace(std::move(value));
    }
    condition_.notify_one();
    return true;
  }

  template <typename Rep, typename Period>
  std::optional<T> wait_pop(
      const std::chrono::duration<Rep, Period>& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait_for(lock, timeout,
                        [this] { return value_.has_value() || closed_; });
    if (!value_.has_value()) {
      return std::nullopt;
    }

    auto result = std::move(value_);
    value_.reset();
    return result;
  }

  void close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return;
      }
      closed_ = true;
    }
    condition_.notify_all();
  }

  bool closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<T> value_;
  bool closed_ = false;
};

}  // namespace rppg_qnn
