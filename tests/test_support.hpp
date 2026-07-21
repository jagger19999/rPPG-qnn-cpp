#pragma once

#include <iostream>

namespace test_support {

inline int failures = 0;

inline void expect_true(bool condition, const char* expression, const char* file, int line) {
  if (!condition) {
    std::cerr << file << ':' << line << ": EXPECT_TRUE(" << expression << ") failed\n";
    ++failures;
  }
}

template <typename Actual, typename Expected>
inline void expect_eq(const Actual& actual,
                      const Expected& expected,
                      const char* actual_expression,
                      const char* expected_expression,
                      const char* file,
                      int line) {
  if (!(actual == expected)) {
    std::cerr << file << ':' << line << ": EXPECT_EQ(" << actual_expression << ", "
              << expected_expression << ") failed\n";
    ++failures;
  }
}

inline int finish() {
  return failures == 0 ? 0 : 1;
}

}  // namespace test_support

#define EXPECT_TRUE(expression) \
  ::test_support::expect_true((expression), #expression, __FILE__, __LINE__)

#define EXPECT_EQ(actual, expected) \
  ::test_support::expect_eq((actual), (expected), #actual, #expected, __FILE__, __LINE__)
