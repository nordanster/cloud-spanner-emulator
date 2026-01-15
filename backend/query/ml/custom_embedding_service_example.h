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

#ifndef THIRD_PARTY_CLOUD_SPANNER_EMULATOR_BACKEND_QUERY_ML_CUSTOM_EMBEDDING_SERVICE_EXAMPLE_H_
#define THIRD_PARTY_CLOUD_SPANNER_EMULATOR_BACKEND_QUERY_ML_CUSTOM_EMBEDDING_SERVICE_EXAMPLE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "backend/query/ml/embedding_service_interface.h"

namespace google::spanner::emulator::backend {

// Example: Ollama embedding service implementation
// Endpoint format: local://localhost:11434/ollama/embed
//
// Ollama API format:
//   POST /api/embeddings
//   {"model": "nomic-embed-text", "prompt": "text to embed"}
//   Returns: {"embedding": [0.1, 0.2, ...]}
class OllamaEmbeddingService : public EmbeddingServiceInterface {
 public:
  explicit OllamaEmbeddingService(const EmbeddingConfig& config);

  absl::StatusOr<std::vector<float>> GetEmbedding(
      absl::string_view text) override;

  absl::StatusOr<std::vector<std::vector<float>>> GetEmbeddings(
      const std::vector<std::string>& texts) override;

  std::string GetServiceType() const override { return "ollama"; }

 private:
  EmbeddingConfig config_;
  std::string model_name_ = "nomic-embed-text";  // Default Ollama model
};

// Example: Custom proprietary embedding service
// Endpoint format: local://localhost:8080/custom/embed
//
// Your custom API format:
//   POST /embed
//   {"text": "content", "model": "custom-model-v1"}
//   Returns: {"embeddings": [0.1, 0.2, ...]}
class CustomEmbeddingService : public EmbeddingServiceInterface {
 public:
  explicit CustomEmbeddingService(const EmbeddingConfig& config);

  absl::StatusOr<std::vector<float>> GetEmbedding(
      absl::string_view text) override;

  absl::StatusOr<std::vector<std::vector<float>>> GetEmbeddings(
      const std::vector<std::string>& texts) override;

  std::string GetServiceType() const override { return "CustomEmbeddingService"; }

 private:
  EmbeddingConfig config_;
  std::string model_name_ = "qwen3-embedding";
  std::string api_key_;  // Optional API key for authentication
};

// Factory functions for registering these services
std::unique_ptr<EmbeddingServiceInterface> CreateOllamaService(
    const EmbeddingConfig& config);

std::unique_ptr<EmbeddingServiceInterface> CreateCustomService(
    const EmbeddingConfig& config);

}  // namespace google::spanner::emulator::backend

#endif  // THIRD_PARTY_CLOUD_SPANNER_EMULATOR_BACKEND_QUERY_ML_CUSTOM_EMBEDDING_SERVICE_EXAMPLE_H_
