#include "rppg_qnn/latest_queue.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>
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
    std::atomic<bool> entered_wait{false};
    std::atomic<bool> woke{false};
    std::thread waiter([&] {
      entered_wait.store(true, std::memory_order_release);
      const auto value = queue.wait_pop(5s);
      woke.store(!value.has_value(), std::memory_order_release);
    });

    while (!entered_wait.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    std::this_thread::sleep_for(10ms);
    queue.close();
    waiter.join();
    EXPECT_TRUE(woke.load(std::memory_order_acquire));
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
