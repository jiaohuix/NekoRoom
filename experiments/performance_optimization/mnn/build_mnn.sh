#!/bin/bash
# $ANDROID_NDK = 28.2.13676358
set -euo pipefail
CONFIG_FILE="$(dirname "$0")/performance.env"
if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

# SME2 is a compile-time MNN option.  Runtime CPU feature detection still
# chooses SME2/i8mm/dotprod automatically on one APK; changing this variable
# requires rebuilding and repackaging libMNN.so.
MNN_SME2="${MNN_SME2:-${MNN_SME2_CONFIG:-ON}}"
printf 'ANDROID_NDK=%s MNN_SME2=%s\n' "${ANDROID_NDK:-unset}" "$MNN_SME2"
cmake ../../../ \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_STL=c++_static \
-DMNN_USE_LOGCAT=true \
-DMNN_BUILD_BENCHMARK=ON \
-DMNN_USE_SSE=OFF \
-DMNN_BUILD_TEST=OFF \
-DANDROID_NATIVE_API_LEVEL=android-28  \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DMNN_LOW_MEMORY=true \
-DMNN_CPU_WEIGHT_DEQUANT_GEMM=true \
-DMNN_BUILD_LLM=true \
-DMNN_SUPPORT_TRANSFORMER_FUSE=true \
-DMNN_ARM82=true \
-DMNN_SME2="$MNN_SME2" \
-DMNN_USE_LOGCAT=true \
-DMNN_OPENCL=false \
-DMNN_VULKAN=false \
-DMNN_SEP_BUILD=ON \
-DMNN_NNAPI=ON \
-DNATIVE_LIBRARY_OUTPUT=. \
-DNATIVE_INCLUDE_OUTPUT=. \
"$@"

make -j8
