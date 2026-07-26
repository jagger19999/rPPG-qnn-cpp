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
onnx_runtime="$root/android/app/src/main/cpp/android_onnx_cpu_runtime.cpp"

for required_file in \
  "$manifest" \
  "$app_gradle" \
  "$gitignore" \
  "$gradlew" \
  "$haar_asset" \
  "$onnx_runtime" \
  "$root/android/gradlew.bat" \
  "$gradle_wrapper_jar" \
  "$gradle_wrapper_properties"; do
  if [[ ! -f "$required_file" ]]; then
    echo "android packaging check: required file is missing: $required_file" >&2
    exit 1
  fi
done

constructor_contract=$(sed -n \
  '/OnnxCpuSession(DeepModel model/,/^  }$/p' "$onnx_runtime")
if ! grep -Fq 'OnnxCpuSession(DeepModel model, const std::string& model_path) try' \
       <<<"$constructor_contract" ||
   ! grep -Fq 'catch (const AppError&) {' <<<"$constructor_contract" ||
   ! grep -Fq 'catch (const Ort::Exception& error) {' <<<"$constructor_contract" ||
   ! grep -Fq 'ErrorCode::ModelLoadFailed' <<<"$constructor_contract" ||
   ! grep -Fq 'ONNX Runtime CPU model load failed:' <<<"$constructor_contract"; then
  echo "android packaging check: ONNX constructor must use a function-try-block and translate Ort initialization errors" >&2
  exit 1
fi

if ! grep -Fq '<opencv_storage>' "$haar_asset" ||
   ! grep -Fq 'type_id="opencv-cascade-classifier"' "$haar_asset" ||
   ! grep -Fq '<featureType>HAAR</featureType>' "$haar_asset"; then
  echo "android packaging check: packaged Haar cascade is invalid" >&2
  exit 1
fi

if [[ ! -f "$root/android/branding/logo.png" ]]; then
  echo "android packaging check: branding logo missing" >&2
  exit 1
fi
if ! grep -Fq 'android:icon="@mipmap/ic_launcher"' "$manifest"; then
  echo "android packaging check: application icon must be @mipmap/ic_launcher" >&2
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
   [[ $(grep -Fc 'android.permission.BLUETOOTH_SCAN' "$manifest" || true) -ne 1 ]] ||
   [[ $(grep -Fc 'android.permission.BLUETOOTH_CONNECT' "$manifest" || true) -ne 1 ]] ||
   [[ $(grep -Fc 'android:usesPermissionFlags="neverForLocation"' "$manifest" || true) -ne 1 ]] ||
   [[ $(grep -Fc 'android.permission.BLUETOOTH"' "$manifest" || true) -ne 1 ]] ||
   [[ $(grep -Fc 'android.permission.BLUETOOTH_ADMIN"' "$manifest" || true) -ne 1 ]] ||
   [[ $(grep -Ec '<[[:space:]]*uses-permission([[:space:]/>]|-)' "$manifest" || true) -ne 5 ]]; then
  echo "android packaging check: CAMERA + BLE scan/connect (+ legacy BT) permissions required" >&2
  exit 1
fi
if [[ $(grep -Fc 'android:name="android.hardware.camera.any"' "$manifest" || true) -ne 1 ]] ||
   [[ $(grep -Fc 'android:name="android.hardware.bluetooth_le"' "$manifest" || true) -ne 1 ]] ||
   [[ $(grep -Fc 'android:required="false"' "$manifest" || true) -ne 2 ]] ||
   [[ $(grep -Ec '<[[:space:]]*uses-feature([[:space:]/>]|$)' "$manifest" || true) -ne 2 ]]; then
  echo "android packaging check: camera.any and bluetooth_le must be the only optional features" >&2
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
android/app/src/main/cpp/android_onnx_model_contract.cpp
android/app/src/main/cpp/android_onnx_model_contract.hpp
android/app/src/main/cpp/android_qnn_preflight_stub.cpp
android/app/src/main/cpp/native_bridge.cpp
android/app/src/main/java/com/jagger/rppgbench/CameraStartGeneration.java
android/app/src/main/java/com/jagger/rppgbench/MainActivity.java
android/app/src/main/java/com/jagger/rppgbench/ModelIntegrity.java
android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java
android/app/src/main/java/com/jagger/rppgbench/PreparedModel.java
android/app/src/main/java/com/jagger/rppgbench/PreviewRotation.java
android/app/src/main/java/com/jagger/rppgbench/ui/FaceBoxOverlay.java
android/app/src/main/java/com/jagger/rppgbench/ui/HrMetricCard.java
android/app/src/main/java/com/jagger/rppgbench/ui/HrStatusFormatter.java
android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformCard.java
android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformGeometry.java
android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformSnapshot.java
android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformState.java
android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformView.java
android/app/src/main/java/com/jagger/rppgbench/watch/AndroidBleBackend.java
android/app/src/main/java/com/jagger/rppgbench/watch/HeartRateParser.java
android/app/src/main/java/com/jagger/rppgbench/watch/WatchAligner.java
android/app/src/main/java/com/jagger/rppgbench/watch/WatchBleBackend.java
android/app/src/main/java/com/jagger/rppgbench/watch/WatchBleWorker.java
android/app/src/main/java/com/jagger/rppgbench/watch/WatchContracts.java
android/app/src/main/java/com/jagger/rppgbench/watch/WatchCsvExport.java
android/app/src/main/java/com/jagger/rppgbench/watch/WatchSampleStore.java
EOF
)
actual_sources=$(CDPATH= cd -- "$root" && find android \
  \( -name .cxx -o -name .externalNativeBuild -o -name .gradle -o -name build -o -path '*/src/test/*' \) \
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
native_bridge_cpp="$root/android/app/src/main/cpp/native_bridge.cpp"
native_bridge_java="$root/android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java"
activity_java="$root/android/app/src/main/java/com/jagger/rppgbench/MainActivity.java"
prepared_model_java="$root/android/app/src/main/java/com/jagger/rppgbench/PreparedModel.java"
strings_xml="$root/android/app/src/main/res/values/strings.xml"

if ! grep -Fq 'Executors.newSingleThreadExecutor' "$activity_java" ||
   ! grep -Fq 'PreparedModel.prepare' "$activity_java" ||
   ! grep -Fq 'cameraStartGeneration.isCurrent' "$activity_java" ||
   ! grep -Fq 'cameraStartGeneration.destroy()' "$activity_java" ||
   ! grep -Fq 'cameraStartExecutor.shutdown()' "$activity_java" ||
   ! grep -Fq 'ParcelFileDescriptor.open' "$prepared_model_java" ||
   ! grep -Fq 'ParcelFileDescriptor.dup' "$prepared_model_java" ||
   ! grep -Fq '"/proc/self/fd/"' "$prepared_model_java"; then
  echo "android packaging check: background start and stable-FD model preparation are required" >&2
  exit 1
fi

for field_mapping in \
  'camera_id|camera ID' \
  'method|traditional method' \
  'cascade_path|cascade path' \
  'output_directory|output directory' \
  'deep_model|deep model' \
  'model_path|model path'; do
  field=${field_mapping%%|*}
  label=${field_mapping#*|}
  if ! grep -Fq "from_jstring(env, $field, \"$label\")" "$native_bridge_cpp"; then
    echo "android packaging check: JNI string field $field must report as $label" >&2
    exit 1
  fi
done

if [[ $(grep -Fc 'deepCard.bind("深度 TSCAN"' "$activity_java" || true) -ne 2 ]] ||
   ! grep -Fq '<string name="deep_selector_label">使用 ONNX Runtime CPU 运行 TSCAN</string>' "$strings_xml" ||
   ! grep -Fq '<string name="camera_smoke_boundary">Camera2 + GREEN/POS/CHROM + 可选 TSCAN ONNX Runtime CPU；不使用 QNN/Adreno。</string>' "$strings_xml" ||
   grep -F 'deepCard.bind(' "$activity_java" | grep -Fq 'EfficientPhys' ||
   grep -E '<string name="(deep_selector_label|camera_smoke_boundary)">' "$strings_xml" |
     grep -Fq 'EfficientPhys'; then
  echo "android packaging check: active deep-model UI must label the runtime as TSCAN" >&2
  exit 1
fi

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
  'public static native void nativeSetPreviewSurface(long handle, android.view.Surface surface);' \
  'public static native void nativeSetDisplayRotation(long handle, int rotationDegrees);' \
  'public static native String nativeStart(long handle);' \
  'public static native String nativeStop(long handle);' \
  'public static native void nativeDestroy(long handle);' \
  'public static native String nativeGetStatus(long handle);' \
  'public static native byte[] nativeGetRoiJpeg(long handle);'; do
  if ! grep -Fq "$required_api" "$native_bridge_java"; then
    echo "android packaging check: NativeBridge must declare $required_api" >&2
    exit 1
  fi
done
if ! grep -Fq 'find_package(OpenCV 4.13.0 EXACT REQUIRED COMPONENTS core imgproc imgcodecs objdetect)' "$camera_cmake" ||
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
  nativeSetPreviewSurface \
  nativeSetDisplayRotation \
  nativeStart \
  nativeStop \
  nativeDestroy \
  TextureView \
  FaceBoxOverlay \
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
android/app/src/main/cpp/android_onnx_model_contract.cpp
android/app/src/main/cpp/android_onnx_model_contract.hpp
android/app/src/main/cpp/android_qnn_preflight_stub.cpp
android/app/src/main/cpp/native_bridge.cpp
android/app/src/main/java/com/jagger/rppgbench/CameraStartGeneration.java
android/app/src/main/java/com/jagger/rppgbench/MainActivity.java
android/app/src/main/java/com/jagger/rppgbench/ModelIntegrity.java
android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java
android/app/src/main/java/com/jagger/rppgbench/PreparedModel.java
android/app/src/main/java/com/jagger/rppgbench/PreviewRotation.java
android/app/src/main/java/com/jagger/rppgbench/ui/FaceBoxOverlay.java
android/app/src/main/java/com/jagger/rppgbench/ui/HrMetricCard.java
android/app/src/main/java/com/jagger/rppgbench/ui/HrStatusFormatter.java
android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformCard.java
android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformGeometry.java
android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformSnapshot.java
android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformState.java
android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformView.java
android/app/src/main/java/com/jagger/rppgbench/watch/AndroidBleBackend.java
android/app/src/main/java/com/jagger/rppgbench/watch/HeartRateParser.java
android/app/src/main/java/com/jagger/rppgbench/watch/WatchAligner.java
android/app/src/main/java/com/jagger/rppgbench/watch/WatchBleBackend.java
android/app/src/main/java/com/jagger/rppgbench/watch/WatchBleWorker.java
android/app/src/main/java/com/jagger/rppgbench/watch/WatchContracts.java
android/app/src/main/java/com/jagger/rppgbench/watch/WatchCsvExport.java
android/app/src/main/java/com/jagger/rppgbench/watch/WatchSampleStore.java
android/app/src/main/res/layout/activity_main.xml
android/app/src/main/res/layout/view_hr_metric_card.xml
android/app/src/main/res/layout/view_ppg_waveform_card.xml
android/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml
android/app/src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml
android/app/src/main/res/mipmap-hdpi/ic_launcher.png
android/app/src/main/res/mipmap-hdpi/ic_launcher_round.png
android/app/src/main/res/mipmap-mdpi/ic_launcher.png
android/app/src/main/res/mipmap-mdpi/ic_launcher_round.png
android/app/src/main/res/mipmap-xhdpi/ic_launcher.png
android/app/src/main/res/mipmap-xhdpi/ic_launcher_round.png
android/app/src/main/res/mipmap-xxhdpi/ic_launcher.png
android/app/src/main/res/mipmap-xxhdpi/ic_launcher_round.png
android/app/src/main/res/mipmap-xxxhdpi/ic_launcher.png
android/app/src/main/res/mipmap-xxxhdpi/ic_launcher_foreground.png
android/app/src/main/res/mipmap-xxxhdpi/ic_launcher_round.png
android/app/src/main/res/values/colors.xml
android/app/src/main/res/values/strings.xml
android/app/src/test/java/com/jagger/rppgbench/CameraStartGenerationTest.java
android/app/src/test/java/com/jagger/rppgbench/ModelIntegrityTest.java
android/app/src/test/java/com/jagger/rppgbench/PreviewRotationTest.java
android/app/src/test/java/com/jagger/rppgbench/ui/HrStatusFormatterTest.java
android/app/src/test/java/com/jagger/rppgbench/ui/PpgWaveformGeometryTest.java
android/app/src/test/java/com/jagger/rppgbench/ui/PpgWaveformSnapshotTest.java
android/app/src/test/java/com/jagger/rppgbench/ui/PpgWaveformStateTest.java
android/app/src/test/java/com/jagger/rppgbench/watch/HeartRateParserTest.java
android/app/src/test/java/com/jagger/rppgbench/watch/WatchAlignerTest.java
android/app/src/test/java/com/jagger/rppgbench/watch/WatchBleWorkerTest.java
android/app/src/test/java/com/jagger/rppgbench/watch/WatchCsvExportTest.java
android/app/src/test/java/com/jagger/rppgbench/watch/WatchSampleStoreTest.java
android/branding/logo.png
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

model_artifact=$(find "$root/android" \
  \( -name build -o -name .gradle -o -name .cxx \) -prune -o \
  -type f \( \
  -iname '*.pth' -o \
  -iname '*.pt' -o \
  -iname '*.onnx' -o \
  -iname '*.dlc' -o \
  -iname '*.bin' \
\) -print -quit)
if [[ -n "$model_artifact" ]]; then
  echo "android packaging check: model artifact is forbidden under android/: $model_artifact" >&2
  exit 1
fi
