//
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef THIRD_PARTY_CLOUD_SPANNER_EMULATOR_BACKEND_QUERY_ML_EMBEDDING_SERVICE_INTERFACE_H_
#define THIRD_PARTY_CLOUD_SPANNER_EMULATOR_BACKEND_QUERY_ML_EMBEDDING_SERVICE_INTERFACE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace google::spanner::emulator::backend {

// Configuration for embedding services.
struct EmbeddingConfig {
  std::string base_url;       // e.g., "http://localhost:30106"
  int timeout_ms = 30000;     // 30 second default timeout
  int embedding_dimensions = 1024;  // Expected dimensions
  bool truncate = true;       // Truncate input text if too long
};

// Abstract interface for embedding service implementations.
// This allows plugging in different embedding services (Ollama, HuggingFace,
// custom services, etc.) while maintaining a consistent interface.
class EmbeddingServiceInterface {
 public:
  virtual ~EmbeddingServiceInterface() = default;

  // Get embedding for a single text input.
  // Returns a vector of floats representing the embedding.
  virtual absl::StatusOr<std::vector<float>> GetEmbedding(
      absl::string_view text) = 0;

  // Get embeddings for multiple text inputs (batch).
  // Returns a vector of embedding vectors.
  virtual absl::StatusOr<std::vector<std::vector<float>>> GetEmbeddings(
      const std::vector<std::string>& texts) = 0;

  // Get the name/type of this embedding service (for logging/debugging).
  virtual std::string GetServiceType() const = 0;
};

// Factory function type for creating embedding services.
using EmbeddingServiceFactory =
    std::function<std::unique_ptr<EmbeddingServiceInterface>(const EmbeddingConfig&)>;

// Registry for custom embedding service implementations.
// Allows users to register their own embedding service types.
class EmbeddingServiceRegistry {
 public:
  // Get the singleton instance.
  static EmbeddingServiceRegistry& GetInstance();

  // Register a custom embedding service type.
  // The type_name is matched against the endpoint URL path.
  // Example: RegisterService("custom", CustomEmbeddingServiceFactory)
  //          Then use endpoint: "local://localhost:8080/custom/embed"
  void RegisterService(absl::string_view type_name,
                      EmbeddingServiceFactory factory);

  // Create an embedding service based on the endpoint URL.
  // Parses the endpoint to determine which service type to create.
  //
  // Supported formats:
  //   local://host:port/embed          -> Default (HuggingFace TEI)
  //   local://host:port/ollama/embed   -> Ollama
  //   local://host:port/custom/embed   -> Custom registered service
  absl::StatusOr<std::unique_ptr<EmbeddingServiceInterface>> CreateService(
      absl::string_view endpoint, const EmbeddingConfig& config);

 private:
  EmbeddingServiceRegistry() = default;

  absl::flat_hash_map<std::string, EmbeddingServiceFactory> factories_;

  // Default factory (HuggingFace TEI format).
  std::unique_ptr<EmbeddingServiceInterface> CreateDefaultService(
      const EmbeddingConfig& config);
};

}  // namespace google::spanner::emulator::backend

#endif  // THIRD_PARTY_CLOUD_SPANNER_EMULATOR_BACKEND_QUERY_ML_EMBEDDING_SERVICE_INTERFACE_H_
