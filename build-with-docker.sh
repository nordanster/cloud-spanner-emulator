#!/bin/bash
# Build Cloud Spanner Emulator with local embedding support using Docker
#
# This script builds the emulator in a Linux container to avoid macOS/zlib issues

set -e

echo "Building Cloud Spanner Emulator in Docker container..."
echo "This may take 10-15 minutes on first run (dependencies need to be compiled)"
echo ""

docker run --rm -v $(pwd):/workspace -w /workspace ubuntu:22.04 bash -c "
set -e

# Install dependencies
echo 'Installing build dependencies...'
apt-get update -qq
apt-get install -y wget g++ python3 git curl unzip openjdk-11-jdk -qq

# Download and install Bazel 6.5.0
echo 'Installing Bazel 6.5.0...'
ARCH=\$(uname -m)
if [ \"\$ARCH\" = \"aarch64\" ] || [ \"\$ARCH\" = \"arm64\" ]; then
  BAZEL_ARCH=\"arm64\"
else
  BAZEL_ARCH=\"x86_64\"
fi
wget -q https://github.com/bazelbuild/bazel/releases/download/6.5.0/bazel-6.5.0-linux-\${BAZEL_ARCH} -O /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel

# Build the target
echo 'Building emulator...'
# Limit parallel jobs to avoid memory exhaustion with large ZetaSQL template files
# Each g++ process can use 4-8GB during compilation, so we limit to 3 jobs for 30GB Docker allocation
bazel build --jobs=3 --local_ram_resources=25600 //binaries:emulator_main

# Copy the binary to the workspace before container exits
echo 'Copying binary to workspace...'
# Find the actual binary path and copy it
BINARY_PATH=\$(bazel info bazel-bin)/binaries/emulator_main
cp \$BINARY_PATH /workspace/emulator_main
chmod +x /workspace/emulator_main

echo ''
echo '✅ Build completed successfully!'
echo ''
echo 'Binary available at: ./emulator_main'
echo ''
echo 'To run the emulator:'
echo '  ./emulator_main \\\'
echo '    --local_embedding_service_url=http://localhost:30106'
echo ''
"
