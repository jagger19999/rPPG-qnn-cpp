cmake_minimum_required(VERSION 3.20)

project(rppg_qnn_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RPPG_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc videoio objdetect)
find_package(Threads REQUIRED)

if(RPPG_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_library(rppg_qnn_sanitizers INTERFACE)
  target_compile_options(rppg_qnn_sanitizers INTERFACE
                         -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(rppg_qnn_sanitizers INTERFACE
                      -fsanitize=address,undefined)
endif()

add_library(rppg_qnn_core src/config.cpp src/contracts.cpp src/deep_runtime.cpp src/deep_window_builder.cpp src/deep_worker.cpp src/error.cpp src/frame_source.cpp src/green_predictor.cpp src/pipeline.cpp src/qnn_preflight.cpp src/result_sink.cpp src/roi_processor.cpp)
target_include_directories(rppg_qnn_core PUBLIC include)
target_link_libraries(rppg_qnn_core PUBLIC ${OpenCV_LIBS} ${CMAKE_DL_LIBS} Threads::Threads)
target_compile_options(rppg_qnn_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(TARGET rppg_qnn_sanitizers)
  target_link_libraries(rppg_qnn_core PUBLIC rppg_qnn_sanitizers)
endif()

add_executable(rppg_qnn_live src/main.cpp)
target_link_libraries(rppg_qnn_live PRIVATE rppg_qnn_core)
target_compile_options(rppg_qnn_live PRIVATE -Wall -Wextra -Wpedantic -Werror)

include(CTest)
if(BUILD_TESTING)
  add_executable(test_contracts tests/test_contracts.cpp)
  target_link_libraries(test_contracts PRIVATE rppg_qnn_core)
  target_include_directories(test_contracts PRIVATE tests)
  target_compile_options(test_contracts PRIVATE -Wall -Wextra -Wpedantic -Werror)

  add_test(NAME contracts COMMAND test_contracts)

  add_executable(test_config tests/test_config.cpp)
  target_link_libraries(test_config PRIVATE rppg_qnn_core)
  target_include_directories(test_config PRIVATE tests)
  target_compile_options(test_config PRIVATE -Wall -Wextra -Wpedantic -Werror)

  add_test(NAME config COMMAND test_config)

  add_executable(test_latest_queue tests/test_latest_queue.cpp)
  target_include_directories(test_latest_queue PRIVATE include tests)
  target_link_libraries(test_latest_queue PRIVATE Threads::Threads)
  target_compile_options(test_latest_queue PRIVATE -Wall -Wextra -Wpedantic -Werror)
  if(TARGET rppg_qnn_sanitizers)
    target_link_libraries(test_latest_queue PRIVATE rppg_qnn_sanitizers)
  endif()

  add_test(NAME latest_queue COMMAND test_latest_queue)

  add_executable(test_frame_source tests/test_frame_source.cpp)
  target_link_libraries(test_frame_source PRIVATE rppg_qnn_core)
  target_include_directories(test_frame_source PRIVATE tests)
  target_compile_options(test_frame_source PRIVATE -Wall -Wextra -Wpedantic -Werror)

  add_test(NAME frame_source COMMAND test_frame_source)

  add_executable(test_roi_processor tests/test_roi_processor.cpp)
  target_link_libraries(test_roi_processor PRIVATE rppg_qnn_core)
  target_include_directories(test_roi_processor PRIVATE tests)
  target_compile_options(test_roi_processor PRIVATE -Wall -Wextra -Wpedantic -Werror)

  add_test(NAME roi_processor COMMAND test_roi_processor)

  add_executable(test_green_predictor tests/test_green_predictor.cpp)
  target_link_libraries(test_green_predictor PRIVATE rppg_qnn_core)
  target_include_directories(test_green_predictor PRIVATE tests)
  target_compile_options(test_green_predictor PRIVATE -Wall -Wextra -Wpedantic -Werror)

  add_test(NAME green_predictor COMMAND test_green_predictor)

  add_executable(test_deep_window_builder tests/test_deep_window_builder.cpp)
  target_link_libraries(test_deep_window_builder PRIVATE rppg_qnn_core)
  target_include_directories(test_deep_window_builder PRIVATE tests)
  target_compile_options(test_deep_window_builder PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME deep_window_builder COMMAND test_deep_window_builder)

  add_executable(test_deep_worker tests/test_deep_worker.cpp)
  target_link_libraries(test_deep_worker PRIVATE rppg_qnn_core)
  target_include_directories(test_deep_worker PRIVATE tests)
  target_compile_options(test_deep_worker PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME deep_worker COMMAND test_deep_worker)
  set_tests_properties(deep_worker PROPERTIES TIMEOUT 30)

  add_library(fake_opencl SHARED tests/test_qnn_preflight.cpp)
  target_compile_definitions(fake_opencl PRIVATE RPPG_FAKE_OPENCL_LIBRARY)
  target_compile_options(fake_opencl PRIVATE -Wall -Wextra -Wpedantic -Werror)

  add_library(fake_dependency SHARED tests/test_qnn_preflight.cpp)
  target_compile_definitions(fake_dependency PRIVATE RPPG_FAKE_DEPENDENCY_LIBRARY)
  target_compile_options(fake_dependency PRIVATE -Wall -Wextra -Wpedantic -Werror)

  add_library(fake_primary SHARED tests/test_qnn_preflight.cpp)
  target_compile_definitions(fake_primary PRIVATE RPPG_FAKE_PRIMARY_LIBRARY)
  target_compile_options(fake_primary PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(fake_primary PRIVATE fake_dependency)

  add_executable(test_qnn_preflight tests/test_qnn_preflight.cpp)
  target_link_libraries(test_qnn_preflight PRIVATE rppg_qnn_core)
  target_include_directories(test_qnn_preflight PRIVATE tests)
  target_compile_definitions(
      test_qnn_preflight PRIVATE
      RPPG_FAKE_OPENCL_LIBRARY_PATH="$<TARGET_FILE:fake_opencl>"
      RPPG_FAKE_PRIMARY_LIBRARY_PATH="$<TARGET_FILE:fake_primary>"
      RPPG_FAKE_DEPENDENCY_LIBRARY_PATH="$<TARGET_FILE:fake_dependency>")
  target_compile_options(test_qnn_preflight PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_qnn_preflight fake_opencl fake_primary fake_dependency)
  add_test(NAME qnn_preflight COMMAND test_qnn_preflight)

  add_executable(test_result_sink tests/test_result_sink.cpp)
  target_link_libraries(test_result_sink PRIVATE rppg_qnn_core)
  target_include_directories(test_result_sink PRIVATE tests)
  target_compile_options(test_result_sink PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME result_sink COMMAND test_result_sink)

  add_executable(test_pipeline tests/test_pipeline.cpp)
  target_link_libraries(test_pipeline PRIVATE rppg_qnn_core)
  target_include_directories(test_pipeline PRIVATE tests)
  target_compile_definitions(test_pipeline PRIVATE RPPG_LIVE_PATH="$<TARGET_FILE:rppg_qnn_live>")
  target_compile_options(test_pipeline PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_pipeline rppg_qnn_live)
  add_test(NAME pipeline COMMAND test_pipeline)
  set_tests_properties(pipeline PROPERTIES TIMEOUT 30)
endif()
