#include "rppg_qnn/build_identity.hpp"

#include <jni.h>

#include <exception>
#include <string>

extern "C" JNIEXPORT jstring JNICALL
Java_com_jagger_rppgbench_NativeBridge_nativeBuildIdentity(JNIEnv* env,
                                                            jclass) noexcept {
  try {
    const std::string identity = rppg_qnn::build_identity_text();
    return env->NewStringUTF(identity.c_str());
  } catch (const std::exception& error) {
    const std::string message = "NATIVE_ERROR: " + std::string(error.what());
    return env->NewStringUTF(message.c_str());
  } catch (...) {
    return env->NewStringUTF("NATIVE_ERROR: unknown");
  }
}
