#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "android packaging check: expected repository root argument" >&2
  echo "usage: $0 <repository-root>" >&2
  exit 2
fi

root=$1
manifest="$root/android/app/src/main/AndroidManifest.xml"
app_gradle="$root/android/app/build.gradle"
gitignore="$root/.gitignore"
gradlew="$root/android/gradlew"
gradle_wrapper_jar="$root/android/gradle/wrapper/gradle-wrapper.jar"
gradle_wrapper_properties="$root/android/gradle/wrapper/gradle-wrapper.properties"
haar_asset="$root/android/app/src/main/assets/haarcascade_frontalface_default.xml"

for required_file in \
  "$manifest" \
  "$app_gradle" \
  "$gitignore" \
  "$gradlew" \
  "$haar_asset" \
  "$root/android/gradlew.bat" \
  "$gradle_wrapper_jar" \
  "$gradle_wrapper_properties"; do
  if [[ ! -f "$required_file" ]]; then
    echo "android packaging check: required file is missing: $required_file" >&2
    exit 1
  fi
done

if ! grep -Fq '<opencv_storage>' "$haar_asset" ||
   ! grep -Fq 'type_id="opencv-cascade-classifier"' "$haar_asset" ||
   ! grep -Fq '<featureType>HAAR</featureType>' "$haar_asset"; then
  echo "android packaging check: packaged Haar cascade is invalid" >&2
  exit 1
fi

if [[ ! -x "$gradlew" ]]; then
  echo "android packaging check: $gradlew must be executable" >&2
  exit 1
fi

if ! grep -Fxq \
  'distributionUrl=https\://services.gradle.org/distributions/gradle-9.1.0-bin.zip' \
  "$gradle_wrapper_properties"; then
  echo "android packaging check: Gradle wrapper distribution URL must be the official Gradle 9.1.0 binary distribution" >&2
  exit 1
fi

if ! grep -Fxq \
  'distributionSha256Sum=a17ddd85a26b6a7f5ddb71ff8b05fc5104c0202c6e64782429790c933686c806' \
  "$gradle_wrapper_properties"; then
  echo "android packaging check: Gradle wrapper distribution SHA256 is invalid" >&2
  exit 1
fi

wrapper_jar_sha256=$(shasum -a 256 "$gradle_wrapper_jar" | awk '{print $1}')
if [[ "$wrapper_jar_sha256" != \
  '76805e32c009c0cf0dd5d206bddc9fb22ea42e84db904b764f3047de095493f3' ]]; then
  echo "android packaging check: Gradle wrapper JAR SHA256 is invalid" >&2
  exit 1
fi

if ! grep -Fxq "android/**/.cxx/" "$gitignore"; then
  echo "android packaging check: $gitignore must contain exact rule android/**/.cxx/" >&2
  exit 1
fi

if ! grep -Fxq "android/**/.externalNativeBuild/" "$gitignore"; then
  echo "android packaging check: $gitignore must contain exact rule android/**/.externalNativeBuild/" >&2
  exit 1
fi

abi_filter_references=$(grep -o 'abiFilters' "$app_gradle" || true)
abi_configuration_lines=$(grep -Eic 'abi' "$app_gradle" || true)
if [[ "$abi_filter_references" != 'abiFilters' ]] ||
   [[ "$abi_configuration_lines" -ne 1 ]] ||
   ! grep -Eq "^[[:space:]]*abiFilters[[:space:]]+['\"]arm64-v8a['\"][[:space:]]*$" "$app_gradle"; then
  echo "android packaging check: ABI set must be exactly arm64-v8a" >&2
  exit 1
fi

if grep -Eiq '(^|[^[:alnum:]_-])(armeabi(-v7a)?|x86(_64)?|mips(64)?|riscv64|aarch64|arm64)([^[:alnum:]_-]|$)' "$app_gradle"; then
  echo "android packaging check: non-arm64-v8a ABI token is forbidden" >&2
  exit 1
fi

if ! grep -Fq "28.2.13676358" "$app_gradle"; then
  echo "android packaging check: $app_gradle must pin NDK version 28.2.13676358" >&2
  exit 1
fi

if grep -Eiq '<[[:space:]]*uses-native-library([[:space:]/>])' "$manifest"; then
  echo "android packaging check: Camera2 APK must not declare vendor native libraries" >&2
  exit 1
fi
if [[ $(grep -Fc '<uses-permission android:name="android.permission.CAMERA" />' "$manifest" || true) -ne 1 ]] ||
   [[ $(grep -Ec '<[[:space:]]*uses-permission([[:space:]/>]|-)' "$manifest" || true) -ne 1 ]]; then
  echo "android packaging check: CAMERA must be the only requested permission" >&2
  exit 1
fi
if [[ $(grep -Fc 'android:name="android.hardware.camera.any"' "$manifest" || true) -ne 1 ]] ||
   [[ $(grep -Fc 'android:required="false"' "$manifest" || true) -ne 1 ]] ||
   [[ $(grep -Ec '<[[:space:]]*uses-feature([[:space:]/>]|$)' "$manifest" || true) -ne 1 ]]; then
  echo "android packaging check: camera.any must be the only optional feature" >&2
  exit 1
fi

expected_sources=$(cat <<'EOF'
android/app/src/main/cpp/CMakeLists.txt
android/app/src/main/cpp/android_camera_session.cpp
android/app/src/main/cpp/android_camera_session.hpp
android/app/src/main/cpp/android_jni_handle.cpp
android/app/src/main/cpp/android_jni_handle.hpp
android/app/src/main/cpp/android_onnx_cpu_runtime.cpp
android/app/src/main/cpp/android_onnx_cpu_runtime.hpp
android/app/src/main/cpp/android_qnn_preflight_stub.cpp
android/app/src/main/cpp/native_bridge.cpp
android/app/src/main/java/com/jagger/rppgbench/MainActivity.java
android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java
EOF
)
actual_sources=$(CDPATH= cd -- "$root" && find android \
  \( -name .cxx -o -name .externalNativeBuild -o -name .gradle -o -name build \) \
  -prune -o -type f \( \
    -name 'CMakeLists.txt' -o \
    -name '*.cmake' -o \
    -name '*.c' -o \
    -name '*.cc' -o \
    -name '*.cpp' -o \
    -name '*.cxx' -o \
    -name '*.h' -o \
    -name '*.hh' -o \
    -name '*.hpp' -o \
    -name '*.java' -o \
    -name '*.kt' \
  \) -print | LC_ALL=C sort)
if [[ "$actual_sources" != "$expected_sources" ]]; then
  echo "android packaging check: Android source whitelist mismatch" >&2
  printf 'expected:\n%s\nactual:\n%s\n' "$expected_sources" "$actual_sources" >&2
  exit 1
fi

for source_file in $actual_sources; do
  if grep -Eq '(libQnn|QnnGpu|Qnn[A-Z]|QNN[A-Z_]|[Ff]ake[_ -]?[Dd]eep|--deep[ =]fake)' "$root/$source_file"; then
    echo "android packaging check: Android source contains QNN or fake inference: $source_file" >&2
    exit 1
  fi
done

camera_source="$root/android/app/src/main/cpp/android_camera_session.cpp"
camera_cmake="$root/android/app/src/main/cpp/CMakeLists.txt"
native_bridge_java="$root/android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java"
activity_java="$root/android/app/src/main/java/com/jagger/rppgbench/MainActivity.java"
for required_symbol in \
  ACameraManager_getCameraIdList \
  AImageReader_new \
  AIMAGE_FORMAT_YUV_420_888 \
  AImageReader_acquireLatestImage \
  AImage_getTimestamp \
  ACameraCaptureSession_stopRepeating; do
  if ! grep -Fq "$required_symbol" "$camera_source"; then
    echo "android packaging check: Camera2 source must use $required_symbol" >&2
    exit 1
  fi
done
for required_api in \
  'public static native String nativeListCameras();' \
  'public static native long nativeCreate(String cameraId, int width, int height, int fps);' \
  'public static native String nativeConfigureProcessing(' \
  'public static native String nativeStart(long handle);' \
  'public static native String nativeStop(long handle);' \
  'public static native void nativeDestroy(long handle);' \
  'public static native String nativeGetStatus(long handle);'; do
  if ! grep -Fq "$required_api" "$native_bridge_java"; then
    echo "android packaging check: NativeBridge must declare $required_api" >&2
    exit 1
  fi
done
if ! grep -Fq 'find_package(OpenCV 4.13.0 EXACT REQUIRED COMPONENTS core imgproc objdetect)' "$camera_cmake" ||
   ! grep -Fq 'RPPG_OPENCV_ANDROID_SDK' "$camera_cmake"; then
  echo "android packaging check: Android CMake must pin the OpenCV 4.13.0 SDK" >&2
  exit 1
fi
if ! grep -Fq 'onnxruntime-android-1.27.0' "$app_gradle" ||
   ! grep -Fq 'RPPG_ONNXRUNTIME_ANDROID' "$camera_cmake"; then
  echo "android packaging check: ONNX Runtime Android CPU dependency must be pinned" >&2
  exit 1
fi
import_script="$root/scripts/import_efficientphys_model.sh"
if [[ ! -x "$import_script" ]] ||
   ! grep -Fq 'c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d' "$import_script"; then
  echo "android packaging check: EfficientPhys import helper/hash missing" >&2
  exit 1
fi
if rg --files "$root/android" -g '*.onnx' -g '*.pth' -g '*.bin' |
   grep -q .; then
  echo "android packaging check: model artifacts must remain external" >&2
  exit 1
fi
for lifecycle_token in \
  requestPermissions \
  onRequestPermissionsResult \
  nativeListCameras \
  nativeStart \
  nativeStop \
  nativeDestroy \
  onStop; do
  if ! grep -Fq "$lifecycle_token" "$activity_java"; then
    echo "android packaging check: Activity must implement $lifecycle_token" >&2
    exit 1
  fi
done
for required_library in camera2ndk mediandk android log; do
  if ! grep -Eq "find_library\\([^)]*${required_library}|target_link_libraries\\([^)]*${required_library}" "$camera_cmake"; then
    echo "android packaging check: Android native build must link $required_library" >&2
    exit 1
  fi
done

expected_gradle_configs=$(cat <<'EOF'
android/app/build.gradle
android/build.gradle
android/gradle.properties
android/settings.gradle
EOF
)
actual_gradle_configs=$(CDPATH= cd -- "$root" && find android \
  \( -name .cxx -o -name .externalNativeBuild -o -name .gradle -o -name build \) \
  -prune -o -type f \( \
    -name '*.gradle' -o \
    -name '*.gradle.kts' -o \
    -name 'gradle.properties' \
  \) -print | LC_ALL=C sort)
if [[ "$actual_gradle_configs" != "$expected_gradle_configs" ]]; then
  echo "android packaging check: Gradle configuration whitelist mismatch" >&2
  printf 'expected:\n%s\nactual:\n%s\n' "$expected_gradle_configs" "$actual_gradle_configs" >&2
  exit 1
fi

for gradle_config in $actual_gradle_configs; do
  if grep -Eq '(libQnn|QnnGpu|Qnn[A-Z]|QNN[A-Z_]|[Ff]ake[_ -]?[Dd]eep|--deep[ =]fake|\.(pth|pt|onnx|dlc|bin|aar|jar|apk|aab)([^[:alnum:]_]|$)|loadModel)' "$root/$gradle_config"; then
    echo "android packaging check: Gradle configuration contains QNN, fake, or model packaging: $gradle_config" >&2
    exit 1
  fi
done

expected_tracked=$(cat <<'EOF'
android/app/build.gradle
android/app/proguard-rules.pro
android/app/src/main/AndroidManifest.xml
android/app/src/main/assets/haarcascade_frontalface_default.xml
android/app/src/main/cpp/CMakeLists.txt
android/app/src/main/cpp/android_camera_session.cpp
android/app/src/main/cpp/android_camera_session.hpp
android/app/src/main/cpp/android_jni_handle.cpp
android/app/src/main/cpp/android_jni_handle.hpp
android/app/src/main/cpp/android_onnx_cpu_runtime.cpp
android/app/src/main/cpp/android_onnx_cpu_runtime.hpp
android/app/src/main/cpp/android_qnn_preflight_stub.cpp
android/app/src/main/cpp/native_bridge.cpp
android/app/src/main/java/com/jagger/rppgbench/MainActivity.java
android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java
android/app/src/main/res/values/strings.xml
android/build.gradle
android/gradle.properties
android/gradle/wrapper/gradle-wrapper.jar
android/gradle/wrapper/gradle-wrapper.properties
android/gradlew
android/gradlew.bat
android/settings.gradle
EOF
)
actual_tracked=$(
  git -C "$root" ls-files --cached --others --exclude-standard -- android |
    LC_ALL=C sort
)
if [[ "$actual_tracked" != "$expected_tracked" ]]; then
  echo "android packaging check: Android artifact whitelist mismatch" >&2
  printf 'expected:\n%s\nactual:\n%s\n' "$expected_tracked" "$actual_tracked" >&2
  exit 1
fi

model_artifact=$(find "$root/android" -type f \( \
  -iname '*.pth' -o \
  -iname '*.pt' -o \
  -iname '*.onnx' -o \
  -iname '*.dlc' -o \
  -iname '*.bin' \
\) ! \( \
  -name 'compile_commands.json.bin' -o \
  -path '*/.cxx/*/CMakeFiles/*/CMakeDetermineCompilerABI_CXX.bin' -o \
  -path '*/.cxx/*/configure_fingerprint.bin' -o \
  -path '*/.gradle/*/checksums/*-checksums.bin' -o \
  -path '*/.gradle/*/executionHistory/executionHistory.bin' -o \
  -path '*/.gradle/*/fileChanges/last-build.bin' -o \
  -path '*/.gradle/*/fileHashes/fileHashes.bin' -o \
  -path '*/.gradle/*/fileHashes/resourceHashesCache.bin' -o \
  -path '*/.gradle/buildOutputCleanup/outputFiles.bin' -o \
  -path '*/build/intermediates/desugar_graph/*/graph.bin' -o \
  -path '*/build/tmp/compile*JavaWithJavac/previous-compilation-data.bin' \
\) -print -quit)
if [[ -n "$model_artifact" ]]; then
  echo "android packaging check: model artifact is forbidden under android/: $model_artifact" >&2
  exit 1
fi
