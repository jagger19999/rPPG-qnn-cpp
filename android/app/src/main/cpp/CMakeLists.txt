cmake_minimum_required(VERSION 3.22.1)

project(rppg_qnn_android LANGUAGES CXX)

set(RPPG_TSCAN_ORT_INTRA_OP_THREADS "2" CACHE STRING
    "TSCAN ORT intra-op threads (benchmark-supported: 2, 4, 6)")
set(RPPG_EFFICIENTPHYS_ORT_INTRA_OP_THREADS "2" CACHE STRING
    "EfficientPhys ORT intra-op threads (benchmark-supported: 2, 4, 6)")
set_property(CACHE RPPG_TSCAN_ORT_INTRA_OP_THREADS PROPERTY STRINGS 2 4 6)
set_property(CACHE RPPG_EFFICIENTPHYS_ORT_INTRA_OP_THREADS PROPERTY STRINGS 2 4 6)
foreach(thread_var RPPG_TSCAN_ORT_INTRA_OP_THREADS
                   RPPG_EFFICIENTPHYS_ORT_INTRA_OP_THREADS)
    if(NOT ${${thread_var}} MATCHES "^(2|4|6)$")
        message(FATAL_ERROR "${thread_var} must be 2, 4, or 6")
    endif()
endforeach()

get_filename_component(
    RPPG_REPOSITORY_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../../.."
    ABSOLUTE)

if(NOT DEFINED ENV{RPPG_OPENCV_ANDROID_SDK})
    message(FATAL_ERROR "RPPG_OPENCV_ANDROID_SDK must point to OpenCV-android-sdk")
endif()
set(OpenCV_DIR "$ENV{RPPG_OPENCV_ANDROID_SDK}/sdk/native/jni")
find_package(OpenCV 4.13.0 EXACT REQUIRED COMPONENTS core imgproc imgcodecs objdetect)

if(NOT DEFINED ENV{RPPG_ONNXRUNTIME_ANDROID})
    message(FATAL_ERROR
            "RPPG_ONNXRUNTIME_ANDROID must point to onnxruntime-android-1.27.0")
endif()
set(ONNXRUNTIME_ROOT "$ENV{RPPG_ONNXRUNTIME_ANDROID}")
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/headers/onnxruntime_cxx_api.h" OR
   NOT EXISTS "${ONNXRUNTIME_ROOT}/jni/${ANDROID_ABI}/libonnxruntime.so")
    message(FATAL_ERROR "ONNX Runtime Android 1.27.0 headers/library are missing")
endif()
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(
    onnxruntime
    PROPERTIES
    IMPORTED_LOCATION
    "${ONNXRUNTIME_ROOT}/jni/${ANDROID_ABI}/libonnxruntime.so")

add_library(
    rppg_qnn_android
    SHARED
    android_camera_session.cpp
    android_jni_handle.cpp
    android_onnx_model_contract.cpp
    android_onnx_cpu_runtime.cpp
    android_qnn_preflight_stub.cpp
    native_bridge.cpp
    "${RPPG_REPOSITORY_ROOT}/src/build_identity.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/contracts.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/deep_model.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/deep_runtime.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/deep_stabilizer.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/deep_window_builder.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/deep_worker.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/efficientphys_runtime.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/error.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/pipeline.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/result_sink.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/roi_processor.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/traditional_predictor.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/tscan_postprocessor.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/tscan_preprocessor.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/tscan_runtime.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/waveform_snapshot.cpp"
    "${RPPG_REPOSITORY_ROOT}/src/yuv420.cpp")

target_include_directories(
    rppg_qnn_android
    PRIVATE
    "${RPPG_REPOSITORY_ROOT}/include"
    "${ONNXRUNTIME_ROOT}/headers")

target_compile_features(rppg_qnn_android PRIVATE cxx_std_17)
target_compile_definitions(
    rppg_qnn_android
    PRIVATE
    RPPG_ANDROID_CAMERA2=1
    RPPG_ANDROID_ONNX_CPU=1
    RPPG_TSCAN_ORT_INTRA_OP_THREADS=${RPPG_TSCAN_ORT_INTRA_OP_THREADS}
    RPPG_EFFICIENTPHYS_ORT_INTRA_OP_THREADS=${RPPG_EFFICIENTPHYS_ORT_INTRA_OP_THREADS})
target_compile_options(
    rppg_qnn_android
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror)

find_library(CAMERA2NDK_LIBRARY camera2ndk)
find_library(MEDIANDK_LIBRARY mediandk)
find_library(ANDROID_LIBRARY android)
find_library(DL_LIBRARY dl)
find_library(LOG_LIBRARY log)
target_link_libraries(
    rppg_qnn_android
    PRIVATE
    "${CAMERA2NDK_LIBRARY}"
    "${MEDIANDK_LIBRARY}"
    "${ANDROID_LIBRARY}"
    "${DL_LIBRARY}"
    "${LOG_LIBRARY}"
    ${OpenCV_LIBS}
    onnxruntime)
