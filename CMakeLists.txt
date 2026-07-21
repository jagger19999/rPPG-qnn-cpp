cmake_minimum_required(VERSION 3.20)

project(rppg_qnn_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RPPG_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc videoio objdetect)

if(RPPG_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_library(rppg_qnn_sanitizers INTERFACE)
  target_compile_options(rppg_qnn_sanitizers INTERFACE
                         -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(rppg_qnn_sanitizers INTERFACE
                      -fsanitize=address,undefined)
endif()

add_library(rppg_qnn_core src/config.cpp src/contracts.cpp src/error.cpp src/frame_source.cpp)
target_include_directories(rppg_qnn_core PUBLIC include)
target_link_libraries(rppg_qnn_core PUBLIC ${OpenCV_LIBS} ${CMAKE_DL_LIBS})
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

  find_package(Threads REQUIRED)
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
endif()
