cmake_minimum_required(VERSION 3.22.1)

project(rppg_qnn_android LANGUAGES CXX)

get_filename_component(
    RPPG_REPOSITORY_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../../.."
    ABSOLUTE)

add_library(
    rppg_qnn_android
    SHARED
    native_bridge.cpp
    "${RPPG_REPOSITORY_ROOT}/src/build_identity.cpp")

target_include_directories(
    rppg_qnn_android
    PRIVATE
    "${RPPG_REPOSITORY_ROOT}/include")

target_compile_features(rppg_qnn_android PRIVATE cxx_std_17)
target_compile_options(
    rppg_qnn_android
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror)
