FROM ubuntu:22.04

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy the emulator binary
COPY emulator_main /emulator_main
RUN chmod +x /emulator_main

# Expose standard Spanner emulator ports
EXPOSE 9010 9020

# Default command - can be overridden in docker-compose
ENTRYPOINT ["/emulator_main"]
