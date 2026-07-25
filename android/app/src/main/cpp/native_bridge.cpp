#include "android_jni_handle.hpp"
#include "rppg_qnn/build_identity.hpp"
#include "rppg_qnn/error.hpp"

#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <jni.h>

#include <exception>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace {

jstring make_jstring(JNIEnv* env, const std::string& value) noexcept {
  return env->NewStringUTF(value.c_str());
}

std::string error_text(const std::exception& error) {
  const auto* app_error = dynamic_cast<const rppg_qnn::AppError*>(&error);
  const std::string code =
      app_error == nullptr ? "NATIVE_ERROR"
                           : std::string(rppg_qnn::to_string(app_error->code()));
  return code + ": " + error.what();
}

template <typename Callable>
jstring string_result(JNIEnv* env, Callable&& callable) noexcept {
  try {
    return make_jstring(env, std::invoke(std::forward<Callable>(callable)));
  } catch (const std::exception& error) {
    return make_jstring(env, error_text(error));
  } catch (...) {
    return make_jstring(env, "NATIVE_ERROR: unknown");
  }
}

std::string from_jstring(JNIEnv* env, jstring value) {
  if (value == nullptr) {
    throw rppg_qnn::AppError(rppg_qnn::ErrorCode::ConfigInvalid,
                             "camera ID must not be null");
  }
  const char* utf = env->GetStringUTFChars(value, nullptr);
  if (utf == nullptr) {
    throw rppg_qnn::AppError(rppg_qnn::ErrorCode::ConfigInvalid,
                             "camera ID conversion failed");
  }
  std::string result(utf);
  env->ReleaseStringUTFChars(value, utf);
  return result;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_jagger_rppgbench_NativeBridge_nativeBuildIdentity(JNIEnv* env,
                                                            jclass) noexcept {
  return string_result(env, [] { return rppg_qnn::build_identity_text(); });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_jagger_rppgbench_NativeBridge_nativeListCameras(JNIEnv* env,
                                                          jclass) noexcept {
  return string_result(
      env, [] { return rppg_qnn::android::list_cameras_json(); });
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_jagger_rppgbench_NativeBridge_nativeCreate(
    JNIEnv* env, jclass, jstring camera_id, jint width, jint height,
    jint fps) noexcept {
  try {
    return static_cast<jlong>(rppg_qnn::android::create_camera_session(
        from_jstring(env, camera_id), width, height, fps));
  } catch (const std::exception& error) {
    const std::string message = error_text(error);
    jclass exception_class = env->FindClass("java/lang/IllegalStateException");
    if (exception_class != nullptr) {
      env->ThrowNew(exception_class, message.c_str());
    }
    return 0;
  } catch (...) {
    jclass exception_class = env->FindClass("java/lang/IllegalStateException");
    if (exception_class != nullptr) {
      env->ThrowNew(exception_class, "NATIVE_ERROR: unknown");
    }
    return 0;
  }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_jagger_rppgbench_NativeBridge_nativeConfigureProcessing(
    JNIEnv* env, jclass, jlong handle, jstring method, jstring cascade_path,
    jstring output_directory, jboolean deep_enabled,
    jstring model_path) noexcept {
  return string_result(env, [env, handle, method, cascade_path,
                             output_directory, deep_enabled, model_path] {
    const std::string native_method = from_jstring(env, method);
    const std::string native_cascade = from_jstring(env, cascade_path);
    const std::string native_output = from_jstring(env, output_directory);
    const std::string native_model = from_jstring(env, model_path);
    return rppg_qnn::android::configure_camera_processing(
        handle, native_method, native_cascade, native_output,
        deep_enabled == JNI_TRUE, native_model);
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_jagger_rppgbench_NativeBridge_nativeSetPreviewSurface(
    JNIEnv* env, jclass, jlong handle, jobject surface) noexcept {
  try {
    ::ANativeWindow* window = nullptr;
    if (surface != nullptr) {
      window = ANativeWindow_fromSurface(env, surface);
      if (window == nullptr) {
        throw rppg_qnn::AppError(rppg_qnn::ErrorCode::CameraOpenFailed,
                                 "ANativeWindow_fromSurface returned null");
      }
    }
    rppg_qnn::android::set_camera_preview_surface(handle, window);
  } catch (const std::exception& error) {
    const std::string message = error_text(error);
    jclass exception_class = env->FindClass("java/lang/IllegalStateException");
    if (exception_class != nullptr) {
      env->ThrowNew(exception_class, message.c_str());
    }
  } catch (...) {
    jclass exception_class = env->FindClass("java/lang/IllegalStateException");
    if (exception_class != nullptr) {
      env->ThrowNew(exception_class, "NATIVE_ERROR: unknown");
    }
  }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_jagger_rppgbench_NativeBridge_nativeStart(JNIEnv* env, jclass,
                                                    jlong handle) noexcept {
  return string_result(env, [handle] {
    return rppg_qnn::android::start_camera_session(handle);
  });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_jagger_rppgbench_NativeBridge_nativeStop(JNIEnv* env, jclass,
                                                   jlong handle) noexcept {
  return string_result(env, [handle] {
    return rppg_qnn::android::stop_camera_session(handle);
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_jagger_rppgbench_NativeBridge_nativeDestroy(JNIEnv*, jclass,
                                                      jlong handle) noexcept {
  rppg_qnn::android::destroy_camera_session(handle);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_jagger_rppgbench_NativeBridge_nativeGetStatus(JNIEnv* env, jclass,
                                                        jlong handle) noexcept {
  return string_result(env, [handle] {
    return rppg_qnn::android::camera_session_status_json(handle);
  });
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_jagger_rppgbench_NativeBridge_nativeGetRoiJpeg(JNIEnv* env, jclass,
                                                         jlong handle) noexcept {
  try {
    const std::vector<std::uint8_t> jpeg =
        rppg_qnn::android::camera_session_roi_jpeg(handle);
    jbyteArray result = env->NewByteArray(static_cast<jsize>(jpeg.size()));
    if (result == nullptr) {
      return env->NewByteArray(0);
    }
    if (!jpeg.empty()) {
      env->SetByteArrayRegion(
          result, 0, static_cast<jsize>(jpeg.size()),
          reinterpret_cast<const jbyte*>(jpeg.data()));
    }
    return result;
  } catch (const std::exception& error) {
    jclass exception_class = env->FindClass("java/lang/IllegalStateException");
    if (exception_class != nullptr) {
      env->ThrowNew(exception_class, error_text(error).c_str());
    }
    return env->NewByteArray(0);
  } catch (...) {
    jclass exception_class = env->FindClass("java/lang/IllegalStateException");
    if (exception_class != nullptr) {
      env->ThrowNew(exception_class, "NATIVE_ERROR: unknown");
    }
    return env->NewByteArray(0);
  }
}
