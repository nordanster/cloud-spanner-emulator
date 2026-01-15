#!/bin/bash
# Test local embedding support in Cloud Spanner Emulator
#
# Prerequisites:
# 1. Build the emulator: ./build-with-docker.sh
# 2. Start your dm-service-embedding on port 30106

set -e

echo "🧪 Testing Local Embedding Support"
echo "=================================="
echo ""

# Check if embedding service is running
echo "1. Checking if embedding service is available at localhost:30106..."
if curl -s -f http://localhost:30106/info > /dev/null 2>&1; then
    echo "   ✅ Embedding service is running"
else
    echo "   ❌ Embedding service not found!"
    echo ""
    echo "Please start your dm-service-embedding:"
    echo "  cd /Users/nordan/dev/dm/service/dm-service-embedding"
    echo "  docker run -p 30106:80 ghcr.io/digitalmirrorai/dm-service-embedding:latest"
    exit 1
fi

# Test embedding service directly
echo ""
echo "2. Testing embedding service directly..."
RESPONSE=$(curl -s -X POST http://localhost:30106/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": "What is Cloud Spanner?", "truncate": true}')

if echo "$RESPONSE" | grep -q "\["; then
    EMBEDDING_DIM=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data[0]) if data else 0)")
    echo "   ✅ Service returned embedding with $EMBEDDING_DIM dimensions"
else
    echo "   ❌ Service returned unexpected response:"
    echo "   $RESPONSE"
    exit 1
fi

# Start emulator
echo ""
echo "3. Starting Cloud Spanner Emulator with local embedding support..."
echo "   (Press Ctrl+C to stop)"
echo ""

./bazel-bin/binaries/emulator_main \
  --local_embedding_service_url=http://localhost:30106 \
  --local_embedding_dimensions=1024 \
  --local_embedding_timeout_ms=30000

# Note: The following would require the emulator to be running
# and a PostgreSQL client to connect and test SQL queries.
# This is left as manual testing for the user.
