use hegel_formal_bridge_m25::{
    canonical_input_hash_v1, content_hash, decode_strict_cbor, derive_split_role_key,
    encode_canonical_cbor, generate_errata_vector_report_v1, generate_typed_role_report_v1,
    hex_decode, hex_encode, id_digest_preimage_v1, id_digest_v1, rfc6962_canonical_record_root,
    split_row_rank, split_seed_commitment, typed_input_signature_id, CborValue, FormalWireError,
};
use serde_json::{json, Value};
use std::io::{self, Read};

const REQUEST_ERROR: &str = "REJECT_REPLAY_REQUEST";

fn request_error(message: impl Into<String>) -> FormalWireError {
    FormalWireError {
        code: REQUEST_ERROR,
        message: message.into(),
    }
}

fn required<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a Value, FormalWireError> {
    object
        .get(key)
        .ok_or_else(|| request_error(format!("missing request field {key:?}")))
}

fn string_field<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a str, FormalWireError> {
    required(object, key)?
        .as_str()
        .ok_or_else(|| request_error(format!("request field {key:?} must be a string")))
}

fn u16_field(object: &serde_json::Map<String, Value>, key: &str) -> Result<u16, FormalWireError> {
    let value = required(object, key)?.as_u64().ok_or_else(|| {
        request_error(format!("request field {key:?} must be an unsigned integer"))
    })?;
    u16::try_from(value).map_err(|_| request_error(format!("request field {key:?} exceeds uint16")))
}

fn exact_bytes<const N: usize>(hex: &str, field: &str) -> Result<[u8; N], FormalWireError> {
    let value = hex_decode(hex)?;
    value.try_into().map_err(|value: Vec<u8>| {
        request_error(format!(
            "request field {field:?} must decode to {N} bytes, got {}",
            value.len()
        ))
    })
}

fn cbor_from_transport(value: &Value) -> Result<CborValue, FormalWireError> {
    match value {
        Value::Null => Ok(CborValue::Null),
        Value::Bool(value) => Ok(CborValue::Bool(*value)),
        Value::Number(number) => {
            if let Some(value) = number.as_u64() {
                Ok(CborValue::Unsigned(value))
            } else if let Some(value) = number.as_i64() {
                if value >= 0 {
                    Ok(CborValue::Unsigned(value as u64))
                } else {
                    Ok(CborValue::Negative((-1_i128 - value as i128) as u64))
                }
            } else {
                Err(request_error("JSON floating-point values are forbidden"))
            }
        }
        Value::Array(values) => Ok(CborValue::Array(
            values
                .iter()
                .map(cbor_from_transport)
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Value::Object(object) => {
            if object.len() != 1 || !object.contains_key("bytes_hex") {
                return Err(request_error(
                    "formal byte strings require an exact {\"bytes_hex\":\"...\"} wrapper",
                ));
            }
            let encoded = object["bytes_hex"]
                .as_str()
                .ok_or_else(|| request_error("bytes_hex must be a string"))?;
            Ok(CborValue::Bytes(hex_decode(encoded)?))
        }
        Value::String(_) => Err(request_error(
            "JSON strings cannot enter formal CBOR; use a bytes_hex wrapper",
        )),
    }
}

fn cbor_to_transport(value: &CborValue) -> Value {
    match value {
        CborValue::Unsigned(value) => json!(value),
        CborValue::Negative(argument) => {
            let mathematical = -1_i128 - *argument as i128;
            if mathematical >= i64::MIN as i128 {
                json!(mathematical as i64)
            } else {
                json!({"negative_argument": argument.to_string()})
            }
        }
        CborValue::Bytes(bytes) => json!({ "bytes_hex": hex_encode(bytes) }),
        CborValue::Array(values) => Value::Array(values.iter().map(cbor_to_transport).collect()),
        CborValue::Bool(value) => json!(value),
        CborValue::Null => Value::Null,
    }
}

fn typed_role_report_json(report: hegel_formal_bridge_m25::TypedRoleReport) -> Value {
    let samples: Vec<Value> = report
        .samples
        .into_iter()
        .map(|sample| {
            json!({
                "universe_index": sample.universe_index,
                "input_cbor_hex": hex_encode(&sample.input_cbor),
                "canonical_input_hash_hex": hex_encode(&sample.canonical_input_hash),
                "universe_row_cbor_hex": hex_encode(&sample.universe_row_cbor),
                "universe_leaf_hash_hex": hex_encode(&sample.universe_leaf_hash),
                "truth_row_cbor_hex": hex_encode(&sample.truth_row_cbor),
                "truth_leaf_hash_hex": hex_encode(&sample.truth_leaf_hash),
            })
        })
        .collect();
    json!({
        "role_name": report.role_name,
        "input_signature_id": report.input_signature_id,
        "row_count": report.row_count,
        "samples": samples,
        "universe_two_row_root_hex": hex_encode(&report.universe_two_row_root),
        "truth_two_row_root_hex": hex_encode(&report.truth_two_row_root),
        "universe_root_hex": hex_encode(&report.universe_root),
        "truth_root_hex": hex_encode(&report.truth_root),
    })
}

fn errata_vector_report_json(report: hegel_formal_bridge_m25::ErrataVectorReport) -> Value {
    let objects = report
        .objects
        .into_iter()
        .map(|vector| {
            json!({
                "name": vector.name,
                "schema_name": vector.schema_name,
                "tag": vector.tag,
                "status": vector.status,
                "bytes_hex": vector.bytes.map(|value| hex_encode(&value)),
                "candidate_root_hex": vector.candidate_root.map(|value| hex_encode(&value)),
                "error_code": vector.error_code,
            })
        })
        .collect::<Vec<_>>();
    let record_trees = report
        .record_trees
        .into_iter()
        .map(|vector| {
            json!({
                "name": vector.name,
                "schema_name": vector.schema_name,
                "tag": vector.tag,
                "status": vector.status,
                "record_count": vector.record_count,
                "first_record_cbor_hex": vector.first_record_cbor.map(|value| hex_encode(&value)),
                "root_hex": vector.root.map(|value| hex_encode(&value)),
                "error_code": vector.error_code,
            })
        })
        .collect::<Vec<_>>();
    let guard_errors = report
        .guard_errors
        .into_iter()
        .map(|vector| {
            json!({
                "vector_id": vector.vector_id,
                "error_code": vector.error_code,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "machine_freeze_id": report.machine_freeze_id,
        "vector_schema": report.vector_schema,
        "objects": objects,
        "record_trees": record_trees,
        "guard_errors": guard_errors,
    })
}

fn execute(request: &Value) -> Result<Value, FormalWireError> {
    let object = request
        .as_object()
        .ok_or_else(|| request_error("request must be a JSON object"))?;
    let operation = string_field(object, "op")?;
    match operation {
        "encode" => {
            let formal = cbor_from_transport(required(object, "value")?)?;
            let encoded = encode_canonical_cbor(&formal)?;
            Ok(json!({"ok": true, "op": operation, "cbor_hex": hex_encode(&encoded)}))
        }
        "decode" => {
            let encoded = hex_decode(string_field(object, "cbor_hex")?)?;
            let formal = decode_strict_cbor(&encoded)?;
            Ok(json!({
                "ok": true,
                "op": operation,
                "canonical_cbor_hex": hex_encode(&encoded),
                "value": cbor_to_transport(&formal)
            }))
        }
        "content_hash" => {
            let domain = string_field(object, "domain")?;
            let formal = cbor_from_transport(required(object, "value")?)?;
            let encoded = encode_canonical_cbor(&formal)?;
            let digest = content_hash(domain, &formal)?;
            Ok(json!({
                "ok": true,
                "op": operation,
                "cbor_hex": hex_encode(&encoded),
                "digest_hex": hex_encode(&digest)
            }))
        }
        "rfc6962_root" => {
            let values = required(object, "leaves_hex")?
                .as_array()
                .ok_or_else(|| request_error("leaves_hex must be an array"))?;
            let leaves = values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .ok_or_else(|| request_error("every leaves_hex item must be a string"))
                        .and_then(hex_decode)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(json!({
                "ok": true,
                "op": operation,
                "leaf_count": leaves.len(),
                "root_hex": hex_encode(&rfc6962_canonical_record_root(&leaves)?)
            }))
        }
        "derive_role_key" => {
            let seed =
                exact_bytes::<32>(string_field(object, "master_seed_hex")?, "master_seed_hex")?;
            let role_id = u16_field(object, "role_id")?;
            Ok(json!({
                "ok": true,
                "op": operation,
                "role_key_hex": hex_encode(&derive_split_role_key(&seed, role_id))
            }))
        }
        "row_rank" => {
            let role_key =
                exact_bytes::<32>(string_field(object, "role_key_hex")?, "role_key_hex")?;
            let input_hash = exact_bytes::<32>(
                string_field(object, "canonical_input_hash_hex")?,
                "canonical_input_hash_hex",
            )?;
            let role_id = u16_field(object, "role_id")?;
            let stratum_id = u16_field(object, "stratum_id")?;
            Ok(json!({
                "ok": true,
                "op": operation,
                "rank_hex": hex_encode(&split_row_rank(
                    &role_key,
                    role_id,
                    stratum_id,
                    &input_hash,
                ))
            }))
        }
        "seed_commitment" => {
            let seed =
                exact_bytes::<32>(string_field(object, "master_seed_hex")?, "master_seed_hex")?;
            Ok(json!({
                "ok": true,
                "op": operation,
                "commitment_hex": hex_encode(&split_seed_commitment(&seed))
            }))
        }
        "id_digest" => {
            let machine_id = string_field(object, "machine_id")?;
            Ok(json!({
                "ok": true,
                "op": operation,
                "machine_id": machine_id,
                "preimage_hex": hex_encode(&id_digest_preimage_v1(machine_id)?),
                "digest_hex": hex_encode(&id_digest_v1(machine_id)?),
            }))
        }
        "validate_typed_input" => {
            let encoded = hex_decode(string_field(object, "cbor_hex")?)?;
            let value = decode_strict_cbor(&encoded)?;
            let input_signature_id = typed_input_signature_id(&value)?;
            Ok(json!({
                "ok": true,
                "op": operation,
                "input_signature_id": input_signature_id,
                "canonical_cbor_hex": hex_encode(&encoded),
                "canonical_input_hash_hex": hex_encode(&canonical_input_hash_v1(&value)?),
            }))
        }
        "typed_rows" => {
            let role_id = u16_field(object, "role_id")?;
            let report = generate_typed_role_report_v1(role_id)?;
            let mut rendered = typed_role_report_json(report);
            let rendered_object = rendered
                .as_object_mut()
                .expect("typed role report JSON is an object");
            rendered_object.insert("ok".to_owned(), json!(true));
            rendered_object.insert("op".to_owned(), json!(operation));
            Ok(rendered)
        }
        "errata_vectors" => {
            if object.len() != 1 {
                return Err(request_error(
                    "errata_vectors request must contain exactly the op field",
                ));
            }
            let report = generate_errata_vector_report_v1()?;
            let mut rendered = errata_vector_report_json(report);
            let rendered_object = rendered
                .as_object_mut()
                .expect("errata vector report JSON is an object");
            rendered_object.insert("ok".to_owned(), json!(true));
            rendered_object.insert("op".to_owned(), json!(operation));
            Ok(rendered)
        }
        _ => Err(request_error(format!(
            "unsupported operation {operation:?}"
        ))),
    }
}

fn main() {
    let mut input = String::new();
    let result = io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| request_error(format!("failed to read stdin: {error}")))
        .and_then(|_| {
            serde_json::from_str::<Value>(&input)
                .map_err(|error| request_error(format!("invalid JSON request: {error}")))
        })
        .and_then(|request| execute(&request));

    match result {
        Ok(response) => {
            println!(
                "{}",
                serde_json::to_string(&response).expect("JSON response")
            );
        }
        Err(error) => {
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "ok": false,
                    "error_code": error.code,
                    "error": error.message,
                }))
                .expect("JSON error response")
            );
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn errata_replay_has_exact_request_and_response_contract() {
        let response = execute(&json!({"op": "errata_vectors"})).unwrap();
        let object = response.as_object().unwrap();
        let mut keys = object.keys().map(String::as_str).collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "guard_errors",
                "machine_freeze_id",
                "objects",
                "ok",
                "op",
                "record_trees",
                "vector_schema",
            ]
        );
        assert_eq!(object["ok"], json!(true));
        assert_eq!(object["op"], json!("errata_vectors"));
        assert_eq!(object["objects"].as_array().unwrap().len(), 21);
        assert_eq!(object["record_trees"].as_array().unwrap().len(), 8);
        assert_eq!(object["guard_errors"].as_array().unwrap().len(), 15);

        let compact = serde_json::to_vec(&response).unwrap();
        assert_eq!(compact.len(), 20_308);
        assert_eq!(
            hex_encode(&Sha256::digest(&compact)),
            "9c855290ad9f9a6e3e523107e0162e6e3c363afec09224245c5e35075ad8ab4c"
        );

        let error = execute(&json!({"op": "errata_vectors", "extra": 1})).unwrap_err();
        assert_eq!(error.code, REQUEST_ERROR);
    }
}
