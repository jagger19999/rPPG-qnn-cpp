#include "rppg_qnn/latest_queue.hpp"

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "test_support.hpp"

namespace {

using namespace std::chrono_literals;

struct MoveOnly {
  explicit MoveOnly(int new_value) : value(new_value) {}

  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&) noexcept = default;
  MoveOnly& operator=(MoveOnly&&) noexcept = default;

  int value;
};

}  // namespace

int main() {
  {
    rppg_qnn::LatestQueue<int> queue;
    EXPECT_TRUE(queue.push(1));
    EXPECT_TRUE(queue.push(2));

    const auto value = queue.wait_pop(10ms);
    EXPECT_TRUE(value.has_value());
    EXPECT_EQ(*value, 2);
    EXPECT_TRUE(!queue.wait_pop(1ms).has_value());
  }

  {
    rppg_qnn::LatestQueue<int> queue;
    EXPECT_TRUE(!queue.wait_pop(1ms).has_value());
  }

  {
    rppg_qnn::LatestQueue<int> queue;
    EXPECT_TRUE(!queue.closed());
    queue.close();
    queue.close();
    EXPECT_TRUE(queue.closed());
    EXPECT_TRUE(!queue.push(1));
  }

  {
    rppg_qnn::LatestQueue<int> queue;
    std::promise<void> wait_started;
    auto wait_started_future = wait_started.get_future();
    std::promise<std::optional<int>> result;
    auto result_future = result.get_future();
    std::thread waiter([&] {
      wait_started.set_value();
      result.set_value(queue.wait_pop(5s));
    });

    wait_started_future.wait();
    queue.close();

    const auto completion = result_future.wait_for(250ms);
    EXPECT_EQ(completion, std::future_status::ready);
    if (completion == std::future_status::ready) {
      EXPECT_TRUE(!result_future.get().has_value());
    }
    waiter.join();
  }

  {
    rppg_qnn::LatestQueue<int> queue;
    EXPECT_TRUE(queue.push(42));
    queue.close();

    const auto value = queue.wait_pop(1ms);
    EXPECT_TRUE(value.has_value());
    EXPECT_EQ(*value, 42);
    EXPECT_TRUE(!queue.wait_pop(1ms).has_value());
  }

  {
    rppg_qnn::LatestQueue<MoveOnly> queue;
    EXPECT_TRUE(queue.push(MoveOnly{7}));
    const auto value = queue.wait_pop(1ms);
    EXPECT_TRUE(value.has_value());
    EXPECT_EQ(value->value, 7);
  }

  {
    rppg_qnn::LatestQueue<int> queue;
    std::vector<int> received;
    std::thread consumer([&] {
      for (;;) {
        const auto value = queue.wait_pop(10ms);
        if (value.has_value()) {
          received.push_back(*value);
        } else if (queue.closed()) {
          return;
        }
      }
    });

    for (int value = 0; value < 1000; ++value) {
      EXPECT_TRUE(queue.push(value));
    }
    queue.close();
    consumer.join();

    EXPECT_TRUE(!received.empty());
    EXPECT_EQ(received.back(), 999);
    for (std::size_t index = 1; index < received.size(); ++index) {
      EXPECT_TRUE(received[index - 1] < received[index]);
    }
  }

  return test_support::finish();
}
