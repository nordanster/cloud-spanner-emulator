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

#ifndef THIRD_PARTY_CLOUD_SPANNER_EMULATOR_BACKEND_QUERY_ML_EMBEDDING_CLIENT_H_
#define THIRD_PARTY_CLOUD_SPANNER_EMULATOR_BACKEND_QUERY_ML_EMBEDDING_CLIENT_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "backend/query/ml/embedding_service_interface.h"

namespace google::spanner::emulator::backend {

// Client for calling external embedding services compatible with
// Hugging Face Text Embeddings Inference API.
// This is the default implementation used when no service type is specified.
class EmbeddingClient : public EmbeddingServiceInterface {
 public:
  explicit EmbeddingClient(const EmbeddingConfig& config);

  // EmbeddingServiceInterface implementation
  absl::StatusOr<std::vector<float>> GetEmbedding(
      absl::string_view text) override;

  absl::StatusOr<std::vector<std::vector<float>>> GetEmbeddings(
      const std::vector<std::string>& texts) override;

  std::string GetServiceType() const override { return "huggingface-tei"; }

 private:
  EmbeddingConfig config_;

  // Parse JSON response from embedding service.
  // Expected format: [[0.1, 0.2, ...], ...]
  absl::StatusOr<std::vector<std::vector<float>>> ParseEmbeddingResponse(
      const std::string& json_response);
};

}  // namespace google::spanner::emulator::backend

#endif  // THIRD_PARTY_CLOUD_SPANNER_EMULATOR_BACKEND_QUERY_ML_EMBEDDING_CLIENT_H_
