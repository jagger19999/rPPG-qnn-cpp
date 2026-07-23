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

for required_file in "$manifest" "$app_gradle" "$gitignore"; do
  if [[ ! -f "$required_file" ]]; then
    echo "android packaging check: required file is missing: $required_file" >&2
    exit 1
  fi
done

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

if grep -Eiq '<[[:space:]]*(uses-permission([[:space:]/>]|-)|uses-feature([[:space:]/>])|uses-native-library([[:space:]/>]))' "$manifest"; then
  echo "android packaging check: foundation manifest must not declare uses-permission*, uses-feature, or uses-native-library" >&2
  exit 1
fi

expected_sources=$(cat <<'EOF'
android/app/src/main/cpp/CMakeLists.txt
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
  if grep -Eq '(ACamera|AImageReader|CameraManager|CameraDevice|android\.permission\.CAMERA|libQnn|QnnGpu|Qnn[A-Z]|QNN[A-Z_]|[Ff]ake[_ -]?[Dd]eep|--deep[ =]fake|\.(pth|pt|onnx|dlc|bin)([^[:alnum:]_]|$)|loadModel|model[_A-Z]?[Pp]ath)' "$root/$source_file"; then
    echo "android packaging check: foundation source contains Camera, QNN, fake, or model integration: $source_file" >&2
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
  if grep -Eq '(jniLibs|libQnn|QnnGpu|Qnn[A-Z]|QNN[A-Z_]|[Ff]ake[_ -]?[Dd]eep|--deep[ =]fake|\.(pth|pt|onnx|dlc|bin|so|a|aar|jar|apk|aab)([^[:alnum:]_]|$)|loadModel|model[_A-Z]?[Pp]ath)' "$root/$gradle_config"; then
    echo "android packaging check: foundation Gradle configuration contains native binary, QNN, fake, or model packaging: $gradle_config" >&2
    exit 1
  fi
done

expected_tracked=$(cat <<'EOF'
android/app/build.gradle
android/app/proguard-rules.pro
android/app/src/main/AndroidManifest.xml
android/app/src/main/cpp/CMakeLists.txt
android/app/src/main/cpp/native_bridge.cpp
android/app/src/main/java/com/jagger/rppgbench/MainActivity.java
android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java
android/app/src/main/res/values/strings.xml
android/build.gradle
android/gradle.properties
android/settings.gradle
EOF
)
actual_tracked=$(git -C "$root" ls-files -- android | LC_ALL=C sort)
if [[ "$actual_tracked" != "$expected_tracked" ]]; then
  echo "android packaging check: tracked Android artifact whitelist mismatch" >&2
  printf 'expected:\n%s\nactual:\n%s\n' "$expected_tracked" "$actual_tracked" >&2
  exit 1
fi

model_artifact=$(find "$root/android" -type f \( \
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
