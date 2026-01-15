#!/bin/bash
# Build custom Spanner Emulator Docker image with embedding support

set -e

echo "Building custom Cloud Spanner Emulator image with embedding support..."

# Build the image
docker build -t spanner-emulator-embeddings:latest .

echo ""
echo "✅ Image built successfully: spanner-emulator-embeddings:latest"
echo ""
echo "Next steps:"
echo "1. Update your compose.yml to use this image"
echo "2. See the example configuration in compose-changes.txt"
