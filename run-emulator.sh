#!/bin/bash
# Run the Cloud Spanner Emulator in Docker with local embedding support

echo "Starting Cloud Spanner Emulator..."
echo "Emulator will be available on ports 9010 (gRPC) and 9020 (HTTP)"
echo ""

# On macOS, use host.docker.internal to reach services on the host
# Replace localhost with host.docker.internal for the embedding service URL
EMBEDDING_URL="${EMBEDDING_URL:-http://host.docker.internal:30106}"

docker run --rm -it \
  -v $(pwd)/emulator_main:/emulator_main \
  -p 9010:9010 \
  -p 9020:9020 \
  ubuntu:22.04 \
  /emulator_main --local_embedding_service_url="$EMBEDDING_URL"
