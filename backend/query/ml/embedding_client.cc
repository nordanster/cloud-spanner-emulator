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

#include "backend/query/ml/embedding_client.h"

#include <memory>
#include <string>
#include <vector>

#include "zetasql/public/json_value.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "common/errors.h"
#include "httplib.h"
#include "zetasql/base/ret_check.h"
#include "zetasql/base/status_macros.h"

namespace google::spanner::emulator::backend {

EmbeddingClient::EmbeddingClient(const EmbeddingConfig& config)
    : config_(config) {}

absl::StatusOr<std::vector<float>> EmbeddingClient::GetEmbedding(
    absl::string_view text) {
  std::vector<std::string> texts = {std::string(text)};
  ZETASQL_ASSIGN_OR_RETURN(auto embeddings, GetEmbeddings(texts));

  if (embeddings.empty()) {
    return absl::InternalError("No embeddings returned from service");
  }

  return embeddings[0];
}

absl::StatusOr<std::vector<std::vector<float>>> EmbeddingClient::GetEmbeddings(
    const std::vector<std::string>& texts) {

  if (texts.empty()) {
    return absl::InvalidArgumentError("No texts provided for embedding");
  }

  // Parse base URL to extract host and port
  // Expected format: "http://localhost:30106" or "http://localhost:30106/embed"
  std::string url = config_.base_url;

  // Remove http:// or https:// prefix
  std::string host_port = url;
  if (absl::StartsWith(url, "http://")) {
    host_port = url.substr(7);
  } else if (absl::StartsWith(url, "https://")) {
    host_port = url.substr(8);
  }

  // Split host and port
  size_t colon_pos = host_port.find(':');
  std::string host;
  int port = 80;
  std::string path = "/embed";

  if (colon_pos != std::string::npos) {
    host = host_port.substr(0, colon_pos);
    size_t slash_pos = host_port.find('/', colon_pos);
    if (slash_pos != std::string::npos) {
      port = std::stoi(host_port.substr(colon_pos + 1, slash_pos - colon_pos - 1));
      path = host_port.substr(slash_pos);
    } else {
      port = std::stoi(host_port.substr(colon_pos + 1));
    }
  } else {
    size_t slash_pos = host_port.find('/');
    if (slash_pos != std::string::npos) {
      host = host_port.substr(0, slash_pos);
      path = host_port.substr(slash_pos);
    } else {
      host = host_port;
    }
  }

  // Create HTTP client
  httplib::Client client(host, port);
  client.set_connection_timeout(0, config_.timeout_ms * 1000);  // seconds, microseconds
  client.set_read_timeout(config_.timeout_ms / 1000, (config_.timeout_ms % 1000) * 1000);
  client.set_write_timeout(config_.timeout_ms / 1000, (config_.timeout_ms % 1000) * 1000);

  // Build JSON request
  std::string json_request;
  if (texts.size() == 1) {
    // Single input format: {"inputs": "text", "truncate": true}
    json_request = absl::StrFormat(R"({"inputs": "%s", "truncate": %s})",
                                   texts[0],
                                   config_.truncate ? "true" : "false");
  } else {
    // Multiple inputs format: {"inputs": ["text1", "text2"], "truncate": true}
    std::string inputs_array = "[";
    for (size_t i = 0; i < texts.size(); ++i) {
      if (i > 0) inputs_array += ", ";
      inputs_array += absl::StrFormat(R"("%s")", texts[i]);
    }
    inputs_array += "]";
    json_request = absl::StrFormat(R"({"inputs": %s, "truncate": %s})",
                                   inputs_array,
                                   config_.truncate ? "true" : "false");
  }

  // Make POST request
  httplib::Headers headers = {
    {"Content-Type", "application/json"},
  };

  auto response = client.Post(path.c_str(), headers, json_request, "application/json");

  // Check for connection errors
  if (!response) {
    return error::LocalEmbeddingServiceUnavailable(config_.base_url);
  }

  // Check HTTP status code
  if (response->status != 200) {
    return absl::UnavailableError(
        absl::StrCat("Embedding service returned error: HTTP ",
                     response->status, " - ", response->body));
  }

  // Parse JSON response
  return ParseEmbeddingResponse(response->body);
}

absl::StatusOr<std::vector<std::vector<float>>>
EmbeddingClient::ParseEmbeddingResponse(const std::string& json_response) {
  ZETASQL_ASSIGN_OR_RETURN(zetasql::JSONValue json,
                   zetasql::JSONValue::ParseJSONString(json_response));
  zetasql::JSONValueConstRef ref = json.GetConstRef();

  if (!ref.IsArray()) {
    return error::LocalEmbeddingInvalidResponse(
        "Response must be an array of embeddings");
  }

  if (ref.GetArraySize() == 0) {
    return error::LocalEmbeddingInvalidResponse("Response array is empty");
  }

  std::vector<std::vector<float>> result;
  result.reserve(ref.GetArraySize());

  for (uint64_t i = 0; i < ref.GetArraySize(); ++i) {
    zetasql::JSONValueConstRef embedding = ref.GetArrayElement(i);

    if (!embedding.IsArray()) {
      return error::LocalEmbeddingInvalidResponse(
          "Each embedding must be an array of floats");
    }

    std::vector<float> embedding_vec;
    embedding_vec.reserve(embedding.GetArraySize());

    for (uint64_t j = 0; j < embedding.GetArraySize(); ++j) {
      zetasql::JSONValueConstRef elem = embedding.GetArrayElement(j);

      if (!elem.IsNumber()) {
        return error::LocalEmbeddingInvalidResponse(
            "Embedding elements must be numbers");
      }

      embedding_vec.push_back(static_cast<float>(elem.GetDouble()));
    }

    // Validate dimension if configured
    if (config_.embedding_dimensions > 0 &&
        embedding_vec.size() != static_cast<size_t>(config_.embedding_dimensions)) {
      return error::LocalEmbeddingDimensionMismatch(
          config_.embedding_dimensions, embedding_vec.size());
    }

    result.push_back(std::move(embedding_vec));
  }

  return result;
}

}  // namespace google::spanner::emulator::backend
