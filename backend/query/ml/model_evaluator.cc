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

#include "backend/query/ml/model_evaluator.h"

#include <cstdint>
#include <vector>

#include "zetasql/public/catalog.h"
#include "zetasql/public/json_value.h"
#include "zetasql/public/types/struct_type.h"
#include "zetasql/public/types/type.h"
#include "zetasql/public/value.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "backend/common/case.h"
#include "backend/query/ml/embedding_client.h"
#include "backend/query/queryable_model.h"
#include "common/config.h"
#include "farmhash.h"
#include "zetasql/base/ret_check.h"
#include "zetasql/base/status_macros.h"

namespace google::spanner::emulator::backend {

namespace {

constexpr absl::string_view kLocalEndpointPrefix = "local://";

// Detects if an endpoint is a local endpoint (starts with "local://").
bool IsLocalEndpoint(absl::string_view endpoint) {
  return absl::StartsWith(endpoint, kLocalEndpointPrefix);
}

}  // namespace

absl::StatusOr<uint64_t> Fingerprint(const zetasql::Value& value) {
  if (value.is_null()) {
    return 0;
  }

  switch (value.type_kind()) {
    case zetasql::TYPE_INT32:
      // Integers are encoded as strings on the wire.
      return farmhash::Fingerprint64(absl::StrCat(value.int32_value()));
    case zetasql::TYPE_INT64:
      // Integers are encoded as strings on the wire.
      return farmhash::Fingerprint64(absl::StrCat(value.int64_value()));
    case zetasql::TYPE_UINT32:
      // Integers are encoded as strings on the wire.
      return farmhash::Fingerprint64(absl::StrCat(value.uint32_value()));
    case zetasql::TYPE_UINT64:
      // Integers are encoded as strings on the wire.
      return farmhash::Fingerprint64(absl::StrCat(value.uint64_value()));
    case zetasql::TYPE_BOOL:
      return farmhash::Fingerprint(value.bool_value());
    case zetasql::TYPE_FLOAT:
      return farmhash::Fingerprint(value.float_value());
    case zetasql::TYPE_DOUBLE:
      return farmhash::Fingerprint(value.double_value());
    case zetasql::TYPE_STRING:
      return farmhash::Fingerprint64(value.string_value());
    case zetasql::TYPE_BYTES:
      return farmhash::Fingerprint64(value.bytes_value());
    case zetasql::TYPE_ARRAY: {
      uint64_t result = 0;
      for (const zetasql::Value& element : value.elements()) {
        ZETASQL_ASSIGN_OR_RETURN(uint64_t element_hash, Fingerprint(element));
        result += element_hash;
      }
      return result;
    }
    case zetasql::TYPE_STRUCT: {
      uint64_t result = 0;
      for (const zetasql::Value& field : value.fields()) {
        ZETASQL_ASSIGN_OR_RETURN(uint64_t field_hash, Fingerprint(field));
        result += field_hash;
      }
      return result;
    }

    default:
      return absl::UnimplementedError(
          absl::StrCat("ML.PREDICT function does not support inputs of type: ",
                       value.type()->TypeName(zetasql::PRODUCT_EXTERNAL,
                                              /*use_external_float32=*/true)));
  }
}

absl::StatusOr<uint64_t> Fingerprint(const zetasql::JSONValueConstRef& json) {
  if (json.IsNumber()) {
    return farmhash::Fingerprint(json.GetDouble());
  } else if (json.IsString()) {
    return farmhash::Fingerprint64(json.GetString());
  } else if (json.IsBoolean()) {
    return farmhash::Fingerprint(json.GetBoolean());
  } else if (json.IsNull()) {
    return 0;
  } else if (json.IsArray()) {
    uint64_t result = 0;
    for (uint64_t i = 0; i < json.GetArraySize(); ++i) {
      ZETASQL_ASSIGN_OR_RETURN(uint64_t element_hash,
                       Fingerprint(json.GetArrayElement(i)));
      result += element_hash;
    }
    return result;
  } else if (json.IsObject()) {
    uint64_t result = 0;
    for (const auto& [key, value] : json.GetMembers()) {
      ZETASQL_ASSIGN_OR_RETURN(uint64_t field_hash, Fingerprint(value));
      result += field_hash;
    }
    return result;
  } else {
    ZETASQL_RET_CHECK_FAIL() << "Unexpected JSON value type";
  }
}

absl::StatusOr<zetasql::Value> ToValue(uint64_t fingerprint,
                                         const zetasql::Type* type) {
  switch (type->kind()) {
    case zetasql::TYPE_INT32:
      return zetasql::Value::Int32(fingerprint);
    case zetasql::TYPE_INT64:
      return zetasql::Value::Int64(fingerprint);
    case zetasql::TYPE_BOOL:
      return zetasql::Value::Bool(fingerprint % 2 == 0);
    case zetasql::TYPE_FLOAT:
      return zetasql::Value::Float(fingerprint);
    case zetasql::TYPE_DOUBLE:
      return zetasql::Value::Double(fingerprint);
    case zetasql::TYPE_STRING:
      return zetasql::Value::String(absl::StrCat(fingerprint));
    case zetasql::TYPE_BYTES:
      return zetasql::Value::Bytes(absl::StrCat(fingerprint));
    case zetasql::TYPE_ARRAY: {
      ZETASQL_ASSIGN_OR_RETURN(zetasql::Value element_value,
                       ToValue(fingerprint, type->AsArray()->element_type()));
      return zetasql::Value::MakeArray(type->AsArray(), {element_value});
    }
    case zetasql::TYPE_STRUCT: {
      std::vector<zetasql::Value> field_values;
      for (const zetasql::StructField& field : type->AsStruct()->fields()) {
        ZETASQL_ASSIGN_OR_RETURN(zetasql::Value field_value,
                         ToValue(fingerprint, field.type));
        field_values.push_back(field_value);
      }
      return zetasql::Value::MakeStruct(type->AsStruct(), field_values);
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("ML.PREDICT function does not support outputs of type: ",
                       type->TypeName(zetasql::PRODUCT_EXTERNAL,
                                      /*use_external_float32=*/true)));
  }
}

// Default prediction implementation. Takes a fingerprint of all model input
// values, then fills out model output columns by casting the hash value to
// output column type.
absl::Status ModelEvaluator::DefaultPredict(
    const zetasql::Model* model,
    const CaseInsensitiveStringMap<const ModelColumn>& model_inputs,
    CaseInsensitiveStringMap<ModelColumn>& model_outputs) {
  uint64_t input_hash = 0;
  for (const auto& model_input : model_inputs) {
    ZETASQL_ASSIGN_OR_RETURN(uint64_t column_hash,
                     Fingerprint(*model_input.second.value));
    input_hash += column_hash;
  }

  for (const auto& model_output : model_outputs) {
    ZETASQL_ASSIGN_OR_RETURN(
        *model_output.second.value,
        ToValue(input_hash, model_output.second.model_column->GetType()));
  }

  return absl::OkStatus();
}

absl::Status ModelEvaluator::Predict(
    const zetasql::Model* model,
    const CaseInsensitiveStringMap<const ModelColumn>& model_inputs,
    CaseInsensitiveStringMap<ModelColumn>& model_outputs) {
  // Cast to QueryableModel to access backend model properties
  const auto* queryable_model = dynamic_cast<const QueryableModel*>(model);
  if (queryable_model != nullptr) {
    const auto* backend_model = queryable_model->GetWrappedModel();
    if (backend_model != nullptr && backend_model->endpoint().has_value()) {
      const std::string& endpoint = backend_model->endpoint().value();

      // Check if this is a local endpoint
      if (IsLocalEndpoint(endpoint)) {
        return LocalPredict(endpoint, model_inputs, model_outputs);
      }
    }
  }

  // Custom model prediction logic can be added here.
  return DefaultPredict(model, model_inputs, model_outputs);
}

// Default prediction implementation for PG. Takes a fingerprint of all model
// input and produces a single "Outcome" boolean field.
absl::Status DefaultPgPredict(absl::string_view endpoint,
                              const zetasql::JSONValueConstRef& instance,
                              const zetasql::JSONValueConstRef& parameters,
                              zetasql::JSONValueRef prediction) {
  ZETASQL_ASSIGN_OR_RETURN(uint64_t input_hash, Fingerprint(instance));
  prediction.SetToEmptyObject();
  prediction.GetMember("Outcome").SetBoolean(input_hash % 2 == 0);
  return absl::OkStatus();
}

absl::Status ModelEvaluator::PgPredict(
    absl::string_view endpoint, const zetasql::JSONValueConstRef& instance,
    const zetasql::JSONValueConstRef& parameters,
    zetasql::JSONValueRef prediction) {
  // Check if this is a local embedding endpoint
  if (IsLocalEndpoint(endpoint)) {
    return LocalPgPredict(endpoint, instance, parameters, prediction);
  }

  // Custom model prediction logic can be added here.
  return DefaultPgPredict(endpoint, instance, parameters, prediction);
}

absl::Status ModelEvaluator::LocalPredict(
    absl::string_view endpoint,
    const CaseInsensitiveStringMap<const ModelColumn>& model_inputs,
    CaseInsensitiveStringMap<ModelColumn>& model_outputs) {
  // Create embedding client configuration from flags
  EmbeddingConfig config;
  config.base_url = config::local_embedding_service_url();
  config.embedding_dimensions = config::local_embedding_dimensions();
  config.timeout_ms = config::local_embedding_timeout_ms();

  EmbeddingClient client(config);

  // Extract text input from model_inputs
  // Look for a column named "content", "text", or the first STRING column
  std::string input_text;
  bool found_input = false;

  for (const auto& [name, column] : model_inputs) {
    const zetasql::Value* value = column.value;
    if (value->type()->IsString()) {
      input_text = value->string_value();
      found_input = true;
      break;
    }
  }

  if (!found_input) {
    return absl::InvalidArgumentError(
        "No STRING input column found for embedding generation");
  }

  // Get embedding from service
  ZETASQL_ASSIGN_OR_RETURN(std::vector<float> embedding, client.GetEmbedding(input_text));

  // Convert to ARRAY<FLOAT32> and set in model_outputs
  for (auto& [name, column] : model_outputs) {
    const zetasql::Type* output_type = column.model_column->GetType();

    if (output_type->IsArray()) {
      const zetasql::ArrayType* array_type = output_type->AsArray();
      const zetasql::Type* element_type = array_type->element_type();

      if (element_type->IsFloat() || element_type->IsDouble()) {
        // Convert embedding to ZetaSQL values
        std::vector<zetasql::Value> elements;
        elements.reserve(embedding.size());

        for (float f : embedding) {
          if (element_type->IsFloat()) {
            elements.push_back(zetasql::Value::Float(f));
          } else {
            elements.push_back(zetasql::Value::Double(static_cast<double>(f)));
          }
        }

        *column.value = zetasql::Value::MakeArray(array_type, elements).value();
        return absl::OkStatus();
      }
    }
  }

  return absl::InvalidArgumentError(
      "Model output must be ARRAY<FLOAT32> or ARRAY<FLOAT64> for embeddings");
}

absl::Status ModelEvaluator::LocalPgPredict(
    absl::string_view endpoint,
    const zetasql::JSONValueConstRef& instance,
    const zetasql::JSONValueConstRef& parameters,
    zetasql::JSONValueRef prediction) {
  // Create embedding client configuration from flags
  EmbeddingConfig config;
  config.base_url = config::local_embedding_service_url();
  config.embedding_dimensions = config::local_embedding_dimensions();
  config.timeout_ms = config::local_embedding_timeout_ms();

  EmbeddingClient client(config);

  // Extract text input from instance JSON
  std::string input_text;

  if (instance.IsString()) {
    input_text = instance.GetString();
  } else if (instance.IsObject()) {
    // Look for "content", "text", or first string member
    for (const auto& [key, value] : instance.GetMembers()) {
      if (value.IsString()) {
        input_text = value.GetString();
        break;
      }
    }
  }

  if (input_text.empty()) {
    return absl::InvalidArgumentError(
        "No text input found in instance for embedding generation");
  }

  // Get embedding from service
  ZETASQL_ASSIGN_OR_RETURN(std::vector<float> embedding, client.GetEmbedding(input_text));

  // Build prediction result as JSON array
  prediction.SetToEmptyObject();
  zetasql::JSONValueRef embedding_array =
      prediction.GetMember("embedding");
  embedding_array.SetToEmptyArray();

  for (float f : embedding) {
    ZETASQL_RETURN_IF_ERROR(embedding_array.AppendArrayElement(
        zetasql::JSONValue(static_cast<double>(f))));
  }

  return absl::OkStatus();
}

}  // namespace google::spanner::emulator::backend
