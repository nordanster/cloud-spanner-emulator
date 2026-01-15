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

#include "backend/query/ml/custom_embedding_service_example.h"

#include "zetasql/public/json_value.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "common/errors.h"
#include "httplib.h"
#include "zetasql/base/status_macros.h"

namespace google::spanner::emulator::backend {

//==============================================================================
// OllamaEmbeddingService Implementation
//==============================================================================

OllamaEmbeddingService::OllamaEmbeddingService(const EmbeddingConfig& config)
    : config_(config) {}

absl::StatusOr<std::vector<float>> OllamaEmbeddingService::GetEmbedding(
    absl::string_view text) {
  std::vector<std::string> texts = {std::string(text)};
  ZETASQL_ASSIGN_OR_RETURN(auto embeddings, GetEmbeddings(texts));
  if (embeddings.empty()) {
    return absl::InternalError("No embeddings returned from Ollama");
  }
  return embeddings[0];
}

absl::StatusOr<std::vector<std::vector<float>>>
OllamaEmbeddingService::GetEmbeddings(const std::vector<std::string>& texts) {
  // Parse host and port from base URL
  std::string url = config_.base_url;
  std::string host_port = url;
  if (absl::StartsWith(url, "http://")) {
    host_port = url.substr(7);
  } else if (absl::StartsWith(url, "https://")) {
    host_port = url.substr(8);
  }

  size_t colon_pos = host_port.find(':');
  std::string host = host_port.substr(0, colon_pos);
  int port = 11434;  // Ollama default port
  if (colon_pos != std::string::npos) {
    size_t slash_pos = host_port.find('/', colon_pos);
    if (slash_pos != std::string::npos) {
      port = std::stoi(host_port.substr(colon_pos + 1, slash_pos - colon_pos - 1));
    } else {
      port = std::stoi(host_port.substr(colon_pos + 1));
    }
  }

  // Create HTTP client
  httplib::Client client(host, port);
  client.set_connection_timeout(0, config_.timeout_ms * 1000);
  client.set_read_timeout(config_.timeout_ms / 1000, (config_.timeout_ms % 1000) * 1000);

  std::vector<std::vector<float>> results;
  results.reserve(texts.size());

  // Ollama processes one text at a time
  for (const auto& text : texts) {
    // Build Ollama JSON request
    std::string json_request = absl::StrFormat(
        R"({"model": "%s", "prompt": "%s"})", model_name_, text);

    httplib::Headers headers = {{"Content-Type", "application/json"}};
    auto response = client.Post("/api/embeddings", headers, json_request,
                               "application/json");

    if (!response) {
      return error::LocalEmbeddingServiceUnavailable(config_.base_url);
    }

    if (response->status != 200) {
      return absl::UnavailableError(
          absl::StrCat("Ollama returned error: HTTP ", response->status));
    }

    // Parse Ollama response: {"embedding": [0.1, 0.2, ...]}
    ZETASQL_ASSIGN_OR_RETURN(zetasql::JSONValue json,
                     zetasql::JSONValue::ParseJSONString(response->body));
    zetasql::JSONValueConstRef ref = json.GetConstRef();

    if (!ref.IsObject()) {
      return error::LocalEmbeddingInvalidResponse("Response must be an object");
    }

    if (!ref.HasMember("embedding")) {
      return error::LocalEmbeddingInvalidResponse(
          "Response missing 'embedding' field");
    }

    zetasql::JSONValueConstRef embedding_array = ref.GetMember("embedding");
    if (!embedding_array.IsArray()) {
      return error::LocalEmbeddingInvalidResponse(
          "'embedding' must be an array");
    }

    std::vector<float> embedding_vec;
    embedding_vec.reserve(embedding_array.GetArraySize());
    for (uint64_t i = 0; i < embedding_array.GetArraySize(); ++i) {
      zetasql::JSONValueConstRef elem = embedding_array.GetArrayElement(i);
      if (!elem.IsNumber()) {
        return error::LocalEmbeddingInvalidResponse(
            "Embedding elements must be numbers");
      }
      embedding_vec.push_back(static_cast<float>(elem.GetDouble()));
    }

    results.push_back(std::move(embedding_vec));
  }

  return results;
}

//==============================================================================
// DigitalMirrorEmbeddingService Implementation
//==============================================================================

DigitalMirrorEmbeddingService::DigitalMirrorEmbeddingService(
    const EmbeddingConfig& config)
    : config_(config) {
  // Optional: Read API key from environment variable
  const char* api_key_env = std::getenv("DM_EMBEDDING_API_KEY");
  if (api_key_env) {
    api_key_ = api_key_env;
  }
}

absl::StatusOr<std::vector<float>> DigitalMirrorEmbeddingService::GetEmbedding(
    absl::string_view text) {
  std::vector<std::string> texts = {std::string(text)};
  ZETASQL_ASSIGN_OR_RETURN(auto embeddings, GetEmbeddings(texts));
  if (embeddings.empty()) {
    return absl::InternalError("No embeddings returned from DigitalMirror");
  }
  return embeddings[0];
}

absl::StatusOr<std::vector<std::vector<float>>>
DigitalMirrorEmbeddingService::GetEmbeddings(
    const std::vector<std::string>& texts) {
  // Parse host and port from base URL
  std::string url = config_.base_url;
  std::string host_port = url;
  if (absl::StartsWith(url, "http://")) {
    host_port = url.substr(7);
  } else if (absl::StartsWith(url, "https://")) {
    host_port = url.substr(8);
  }

  size_t colon_pos = host_port.find(':');
  std::string host = host_port.substr(0, colon_pos);
  int port = 80;
  if (colon_pos != std::string::npos) {
    size_t slash_pos = host_port.find('/', colon_pos);
    if (slash_pos != std::string::npos) {
      port = std::stoi(host_port.substr(colon_pos + 1, slash_pos - colon_pos - 1));
    } else {
      port = std::stoi(host_port.substr(colon_pos + 1));
    }
  }

  // Create HTTP client
  httplib::Client client(host, port);
  client.set_connection_timeout(0, config_.timeout_ms * 1000);
  client.set_read_timeout(config_.timeout_ms / 1000, (config_.timeout_ms % 1000) * 1000);

  std::vector<std::vector<float>> results;
  results.reserve(texts.size());

  // Your custom API format: {"text": "content", "model": "model-name"}
  for (const auto& text : texts) {
    std::string json_request = absl::StrFormat(
        R"({"text": "%s", "model": "%s"})", text, model_name_);

    httplib::Headers headers = {{"Content-Type", "application/json"}};

    // Add API key if available
    if (!api_key_.empty()) {
      headers.insert({"Authorization", absl::StrCat("Bearer ", api_key_)});
    }

    auto response = client.Post("/embed", headers, json_request,
                               "application/json");

    if (!response) {
      return error::LocalEmbeddingServiceUnavailable(config_.base_url);
    }

    if (response->status != 200) {
      return absl::UnavailableError(
          absl::StrCat("DigitalMirror returned error: HTTP ", response->status,
                      " - ", response->body));
    }

    // Parse custom response format: {"embeddings": [0.1, 0.2, ...]}
    // Or adapt to your actual API format
    ZETASQL_ASSIGN_OR_RETURN(zetasql::JSONValue json,
                     zetasql::JSONValue::ParseJSONString(response->body));
    zetasql::JSONValueConstRef ref = json.GetConstRef();

    // Adjust this based on your actual API response format
    zetasql::JSONValueConstRef embedding_array;
    if (ref.IsArray()) {
      // Format 1: Direct array response [[0.1, 0.2, ...]]
      if (ref.GetArraySize() == 0) {
        return error::LocalEmbeddingInvalidResponse("Empty response array");
      }
      embedding_array = ref.GetArrayElement(0);
    } else if (ref.IsObject() && ref.HasMember("embeddings")) {
      // Format 2: Object with "embeddings" field
      embedding_array = ref.GetMember("embeddings");
    } else if (ref.IsObject() && ref.HasMember("embedding")) {
      // Format 3: Object with "embedding" field (singular)
      embedding_array = ref.GetMember("embedding");
    } else {
      return error::LocalEmbeddingInvalidResponse(
          "Unrecognized response format from DigitalMirror");
    }

    if (!embedding_array.IsArray()) {
      return error::LocalEmbeddingInvalidResponse(
          "Embedding must be an array");
    }

    std::vector<float> embedding_vec;
    embedding_vec.reserve(embedding_array.GetArraySize());
    for (uint64_t i = 0; i < embedding_array.GetArraySize(); ++i) {
      zetasql::JSONValueConstRef elem = embedding_array.GetArrayElement(i);
      if (!elem.IsNumber()) {
        return error::LocalEmbeddingInvalidResponse(
            "Embedding elements must be numbers");
      }
      embedding_vec.push_back(static_cast<float>(elem.GetDouble()));
    }

    results.push_back(std::move(embedding_vec));
  }

  return results;
}

//==============================================================================
// Factory Functions
//==============================================================================

std::unique_ptr<EmbeddingServiceInterface> CreateOllamaService(
    const EmbeddingConfig& config) {
  return std::make_unique<OllamaEmbeddingService>(config);
}

std::unique_ptr<EmbeddingServiceInterface> CreateDigitalMirrorService(
    const EmbeddingConfig& config) {
  return std::make_unique<DigitalMirrorEmbeddingService>(config);
}

}  // namespace google::spanner::emulator::backend
