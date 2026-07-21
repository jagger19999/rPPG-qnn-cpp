#pragma once

#include "rppg_qnn/error.hpp"

#include <exception>
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

template <typename Operation>
inline void expect_app_error(Operation&& operation,
                             rppg_qnn::ErrorCode expected_code,
                             const char* expression,
                             const char* expected_code_expression,
                             const char* file,
                             int line) {
  try {
    operation();
  } catch (const rppg_qnn::AppError& error) {
    if (error.code() != expected_code) {
      std::cerr << file << ':' << line << ": EXPECT_APP_ERROR(" << expression
                << ", " << expected_code_expression
                << ") caught AppError with code "
                << rppg_qnn::to_string(error.code()) << " instead of "
                << rppg_qnn::to_string(expected_code) << '\n';
      ++failures;
    }
    return;
  } catch (const std::exception& error) {
    std::cerr << file << ':' << line << ": EXPECT_APP_ERROR(" << expression
              << ", " << expected_code_expression
              << ") caught a different exception: " << error.what() << '\n';
    ++failures;
    return;
  } catch (...) {
    std::cerr << file << ':' << line << ": EXPECT_APP_ERROR(" << expression
              << ", " << expected_code_expression
              << ") caught a non-standard exception\n";
    ++failures;
    return;
  }

  std::cerr << file << ':' << line << ": EXPECT_APP_ERROR(" << expression
            << ", " << expected_code_expression << ") did not throw\n";
  ++failures;
}

}  // namespace test_support

#define EXPECT_TRUE(expression) \
  ::test_support::expect_true((expression), #expression, __FILE__, __LINE__)

#define EXPECT_EQ(actual, expected) \
  ::test_support::expect_eq((actual), (expected), #actual, #expected, __FILE__, __LINE__)

#define EXPECT_APP_ERROR(expression, expected_code)                         \
  ::test_support::expect_app_error(                                         \
      [&]() { (void)(expression); }, (expected_code), #expression,          \
      #expected_code, __FILE__, __LINE__)
