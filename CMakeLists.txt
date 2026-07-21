cmake_minimum_required(VERSION 3.20)

project(rppg_qnn_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RPPG_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc videoio objdetect)

add_library(rppg_qnn_core src/config.cpp src/contracts.cpp src/error.cpp)
target_include_directories(rppg_qnn_core PUBLIC include)
target_link_libraries(rppg_qnn_core PUBLIC ${OpenCV_LIBS} ${CMAKE_DL_LIBS})
target_compile_options(rppg_qnn_core PRIVATE -Wall -Wextra -Wpedantic -Werror)

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
endif()

if(RPPG_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(rppg_qnn_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(rppg_qnn_core PUBLIC -fsanitize=address,undefined)
endif()
