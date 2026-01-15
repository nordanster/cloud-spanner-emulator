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

#include "backend/query/ml/embedding_service_interface.h"

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "backend/query/ml/embedding_client.h"
#include "common/errors.h"

namespace google::spanner::emulator::backend {

EmbeddingServiceRegistry& EmbeddingServiceRegistry::GetInstance() {
  static EmbeddingServiceRegistry instance;
  return instance;
}

void EmbeddingServiceRegistry::RegisterService(
    absl::string_view type_name, EmbeddingServiceFactory factory) {
  factories_[std::string(type_name)] = std::move(factory);
}

absl::StatusOr<std::unique_ptr<EmbeddingServiceInterface>>
EmbeddingServiceRegistry::CreateService(absl::string_view endpoint,
                                       const EmbeddingConfig& config) {
  // Parse endpoint to extract service type
  // Format: local://host:port/[service_type]/endpoint_path
  // Examples:
  //   local://localhost:30106/embed               -> default (HuggingFace TEI)
  //   local://localhost:11434/ollama/embed        -> ollama
  //   local://localhost:8080/custom/embed         -> custom

  std::string endpoint_str(endpoint);

  // Remove "local://" prefix
  if (absl::StartsWith(endpoint_str, "local://")) {
    endpoint_str = endpoint_str.substr(8);
  }

  // Find the first '/' after the host:port
  size_t first_slash = endpoint_str.find('/');
  if (first_slash == std::string::npos) {
    // No path component, use default
    return CreateDefaultService(config);
  }

  // Extract path after host:port
  std::string path = endpoint_str.substr(first_slash + 1);

  // Check if path starts with a registered service type
  for (const auto& [type_name, factory] : factories_) {
    if (absl::StartsWith(path, type_name + "/") || path == type_name) {
      return factory(config);
    }
  }

  // No registered service type found, use default
  return CreateDefaultService(config);
}

std::unique_ptr<EmbeddingServiceInterface>
EmbeddingServiceRegistry::CreateDefaultService(const EmbeddingConfig& config) {
  // Default to HuggingFace Text Embeddings Inference format
  return std::make_unique<EmbeddingClient>(config);
}

}  // namespace google::spanner::emulator::backend
