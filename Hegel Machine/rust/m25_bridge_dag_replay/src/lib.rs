//! Independent Rust replay for the Phase-3A M2.5 bridge-candidate DAG.
//!
//! Hidden split assignment rows are intentionally absent.  Their six roots
//! and exact counts are accepted only as sealed commitments, and purposes 2/3
//! require a valid purpose-1 Ed25519 signature over the exact bridge root.

use hegel_formal_bridge_m25::{
    content_hash, content_hash_cbor, decode_strict_cbor,
    rfc6962_canonical_record_root, CborValue,
};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

pub const PACKAGE_TAG: u64 = 0x3501;
pub const PACKAGE_SCHEMA_ID: &[u8] = b"hegel-m25-bridge-full-dag-replay-package/1";
pub const PACKAGE_HASH_DOMAIN: &str = "HEGEL/M25/BRIDGE_FULL_DAG_REPLAY_PACKAGE/V1";
pub const SEALED_SPLIT_SCHEMA_ID: &[u8] = b"hegel-sealed-split-commitment/1";
pub const CANONICAL_INPUT_DOMAIN: &str = "HEGEL/CANONICAL_INPUT/V1";

pub const FAIL_PACKAGE_SCHEMA: &str = "FAIL_M25_BRIDGE_REPLAY_PACKAGE_SCHEMA";
pub const FAIL_PACKAGE_AUTHORITY: &str = "FAIL_M25_BRIDGE_REPLAY_AUTHORITY_GUARD";
pub const FAIL_PURPOSE: &str = "FAIL_M25_BRIDGE_REPLAY_PURPOSE";
pub const FAIL_NODE_SET: &str = "FAIL_M25_BRIDGE_REPLAY_NODE_SET";
pub const FAIL_NODE_SCHEMA: &str = "FAIL_M25_BRIDGE_REPLAY_NODE_SCHEMA";
pub const FAIL_NODE_PREIMAGE: &str = "FAIL_M25_BRIDGE_REPLAY_NODE_PREIMAGE";
pub const FAIL_NODE_COUNT: &str = "FAIL_M25_BRIDGE_REPLAY_NODE_COUNT";
pub const FAIL_ROOT_BINDING: &str = "FAIL_M25_BRIDGE_REPLAY_ROOT_BINDING";
pub const FAIL_CANDIDATE: &str = "FAIL_M25_BRIDGE_REPLAY_CANDIDATE";
pub const FAIL_BRIDGE: &str = "FAIL_M25_BRIDGE_REPLAY_BRIDGE";
pub const FAIL_ROLE_BINDING: &str = "FAIL_M25_BRIDGE_REPLAY_CROSS_ROLE";
pub const FAIL_TYPED_BINDING: &str = "FAIL_M25_BRIDGE_REPLAY_TYPED_BINDING";
pub const FAIL_SPLIT_BINDING: &str = "FAIL_M25_BRIDGE_REPLAY_SEALED_SPLIT_BINDING";
pub const FAIL_TRUST_BINDING: &str = "FAIL_M25_BRIDGE_REPLAY_PURPOSE1_TRUST_BINDING";
pub const FAIL_SIGNATURE_PHASE: &str = "FAIL_M25_BRIDGE_REPLAY_SIGNATURE_PHASE";
pub const FAIL_SIGNATURE: &str = "FAIL_M25_BRIDGE_REPLAY_PURPOSE1_SIGNATURE";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayError {
    pub code: &'static str,
    pub detail: String,
}

impl ReplayError {
    fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self { code, detail: detail.into() }
    }
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for ReplayError {}

#[derive(Clone, Copy)]
struct RoleSpec {
    role: u64,
    candidate_index: usize,
    op: u64,
    tag: u64,
    schema: &'static [u8],
    domain: Option<&'static str>,
    count: usize,
    fields: usize,
}

macro_rules! c {
    ($r:expr,$i:expr,$tag:expr,$schema:expr,$domain:expr,$fields:expr) => {
        RoleSpec { role:$r, candidate_index:$i, op:1, tag:$tag, schema:$schema, domain:Some($domain), count:1, fields:$fields }
    };
}
macro_rules! t {
    ($r:expr,$i:expr,$tag:expr,$schema:expr,$count:expr,$fields:expr) => {
        RoleSpec { role:$r, candidate_index:$i, op:2, tag:$tag, schema:$schema, domain:None, count:$count, fields:$fields }
    };
}
macro_rules! s {
    ($r:expr,$i:expr,$count:expr) => {
        RoleSpec { role:$r, candidate_index:$i, op:3, tag:0, schema:SEALED_SPLIT_SCHEMA_ID, domain:None, count:$count, fields:0 }
    };
}

const ROLES: &[RoleSpec] = &[
    c!(1,4,0x3003,b"hegel-dsl-spec/1","HEGEL/DSL_SPEC/V1",19),
    c!(2,5,0x3002,b"hegel-freeze-spec/1","HEGEL/FREEZE_SPEC/V1",9),
    c!(3,6,0x3101,b"hegel-normative-approval-manifest/1","HEGEL/NORMATIVE_APPROVAL_MANIFEST/V1",10),
    c!(4,7,0x3109,b"hegel-dsl-shrink-transition-formal/1","HEGEL/DSL_SHRINK_TRANSITION/V1",21),
    t!(5,8,0x3205,b"hegel-operator-semantics-entry/1",28,8),
    t!(6,9,0x3204,b"hegel-identifier-registry-entry/1",55,7),
    c!(7,10,0x3019,b"hegel-canonical-ast-profile/1","HEGEL/CANONICAL_AST_PROFILE/V1",6),
    c!(8,11,0x301a,b"hegel-canonical-cbor-profile/1","HEGEL/CANONICAL_CBOR_PROFILE/V1",6),
    t!(9,12,0x3206,b"hegel-diagnostic-formal-bridge-record/1",12,11),
    c!(10,13,0x3102,b"hegel-dsl-role-binding-manifest/1","HEGEL/DSL_ROLE_BINDING_MANIFEST/V1",22),
    c!(11,14,0x3102,b"hegel-dsl-role-binding-manifest/1","HEGEL/DSL_ROLE_BINDING_MANIFEST/V1",22),
    c!(12,15,0x3104,b"hegel-split-binding-manifest/1","HEGEL/SPLIT_BINDING_MANIFEST/V1",15),
    c!(13,16,0x3105,b"hegel-custodian-binding-manifest/1","HEGEL/CUSTODIAN_BINDING_MANIFEST/V1",11),
    c!(14,17,0x3106,b"hegel-seed-continuity-manifest/1","HEGEL/SEED_CONTINUITY_MANIFEST/V1",9),
    c!(15,18,0x310d,b"hegel-attestation-bundle/1","HEGEL/ATTESTATION_BUNDLE/V1",1),
    c!(16,19,0x3114,b"hegel-parent-manifest-absence-attestation/2","HEGEL/PARENT_MANIFEST_ABSENCE_ATTESTATION/V2",7),
    c!(17,20,0x3108,b"hegel-hidden-access-ledger-record/1","HEGEL/HIDDEN_ACCESS_LEDGER_RECORD/V1",10),
    c!(18,21,0x3108,b"hegel-hidden-access-ledger-record/1","HEGEL/HIDDEN_ACCESS_LEDGER_RECORD/V1",10),
    c!(19,22,0x3112,b"hegel-opaque-id-registry-snapshot/1","HEGEL/OPAQUE_ID_REGISTRY_SNAPSHOT/V1",5),
    c!(20,23,0x3111,b"hegel-actor-trust-genesis/1","HEGEL/ACTOR_TRUST_GENESIS/V1",5),
    t!(21,24,0x3201,b"hegel-bounded-universe-row/1",480,3),
    t!(22,25,0x3202,b"hegel-target-truth-row/1",480,3),
    t!(23,26,0x3201,b"hegel-bounded-universe-row/1",85,3),
    t!(24,27,0x3202,b"hegel-target-truth-row/1",85,3),
    s!(25,28,192), s!(26,29,96), s!(27,30,192),
    s!(28,31,39), s!(29,32,20), s!(30,33,26),
    c!(31,38,0x300c,b"hegel-implementation-binding/1","HEGEL/IMPLEMENTATION_BINDING/V1",11),
    c!(32,39,0x300c,b"hegel-implementation-binding/1","HEGEL/IMPLEMENTATION_BINDING/V1",11),
    c!(33,40,0x300d,b"hegel-traversal-contract/1","HEGEL/TRAVERSAL_CONTRACT/V1",6),
    c!(34,41,0x300e,b"hegel-bucket-accounting-contract/1","HEGEL/BUCKET_ACCOUNTING_CONTRACT/V1",5),
    c!(35,42,0x300f,b"hegel-program-archive-contract/1","HEGEL/PROGRAM_ARCHIVE_CONTRACT/V1",7),
    c!(36,43,0x3010,b"hegel-output-archive-contract/1","HEGEL/OUTPUT_ARCHIVE_CONTRACT/V1",7),
    c!(37,44,0x3011,b"hegel-m3-state-machine-contract/1","HEGEL/M3_STATE_MACHINE_CONTRACT/V1",6),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayResult {
    pub package_digest: [u8; 32],
    pub candidate_root: [u8; 32],
    pub bridge_root: [u8; 32],
    pub purpose: u64,
    pub purpose1_signature_verified: bool,
    pub authoritative: bool,
}

fn arr<'a>(value: &'a CborValue, code: &'static str, label: &str) -> Result<&'a [CborValue], ReplayError> {
    if let CborValue::Array(items) = value { Ok(items) } else { Err(ReplayError::new(code, format!("{label} is not an array"))) }
}
fn uint(value: &CborValue, code: &'static str, label: &str) -> Result<u64, ReplayError> {
    if let CborValue::Unsigned(v) = value { Ok(*v) } else { Err(ReplayError::new(code, format!("{label} is not uint"))) }
}
fn bytes<'a>(value: &'a CborValue, length: Option<usize>, code: &'static str, label: &str) -> Result<&'a [u8], ReplayError> {
    if let CborValue::Bytes(v) = value {
        if length.is_none_or(|n| n == v.len()) { return Ok(v); }
    }
    Err(ReplayError::new(code, format!("{label} has wrong byte-string shape")))
}
fn root(value: &CborValue, code: &'static str, label: &str) -> Result<[u8; 32], ReplayError> {
    Ok(bytes(value, Some(32), code, label)?.try_into().expect("32 bytes"))
}
fn prefix(value: &CborValue, spec: &RoleSpec) -> Result<(), ReplayError> {
    let fields = arr(value, FAIL_NODE_SCHEMA, "formal preimage")?;
    if fields.len() != 3 + spec.fields
        || fields[0] != CborValue::Unsigned(1)
        || fields[1] != CborValue::Unsigned(spec.tag)
        || fields[2] != CborValue::Bytes(spec.schema.to_vec())
    {
        return Err(ReplayError::new(FAIL_NODE_SCHEMA, format!("role {} formal prefix/length differs", spec.role)));
    }
    Ok(())
}

pub fn bridge_attestation_signature_preimage_v1(
    root: &[u8; 32],
    purpose: u64,
    epoch: u64,
) -> Result<Vec<u8>, ReplayError> {
    if !(1..=3).contains(&purpose) {
        return Err(ReplayError::new(
            FAIL_PURPOSE,
            "bridge signing purpose must be 1, 2, or 3",
        ));
    }
    let mut out = b"HEGEL/BRIDGE_ATTESTATION_SIGNATURE/V1\0".to_vec();
    out.extend_from_slice(root);
    out.extend_from_slice(&(purpose as u16).to_be_bytes());
    out.extend_from_slice(&epoch.to_be_bytes());
    Ok(out)
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn write_new(path: &Path, payload: &[u8]) -> Result<(), ReplayError> {
    let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)
        .map_err(|e| ReplayError::new(FAIL_SIGNATURE, format!("cannot create verifier input: {e}")))?;
    file.set_permissions(fs::Permissions::from_mode(0o600))
        .map_err(|e| ReplayError::new(FAIL_SIGNATURE, format!("cannot set verifier input mode: {e}")))?;
    file.write_all(payload).map_err(|e| ReplayError::new(FAIL_SIGNATURE, format!("cannot write verifier input: {e}")))?;
    file.sync_all().map_err(|e| ReplayError::new(FAIL_SIGNATURE, format!("cannot sync verifier input: {e}")))
}

fn verify_ed25519_openssl(public: &[u8], signature: &[u8], message: &[u8], private_temp_dir: &Path, openssl_path: &Path) -> Result<(), ReplayError> {
    if openssl_path != Path::new("/usr/bin/openssl") || !openssl_path.is_absolute() {
        return Err(ReplayError::new(FAIL_SIGNATURE, "OpenSSL executable must be exactly /usr/bin/openssl"));
    }
    let metadata=fs::symlink_metadata(private_temp_dir)
        .map_err(|e|ReplayError::new(FAIL_SIGNATURE,format!("cannot inspect verifier directory: {e}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() || metadata.permissions().mode() & 0o777 != 0o700 {
        return Err(ReplayError::new(FAIL_SIGNATURE,"verifier directory must be a non-symlink mode-0700 directory"));
    }
    let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let stem=format!("hegel-m25-replay-{}-{}",std::process::id(),nonce);
    let public_path = private_temp_dir.join(format!("{stem}.public.der"));
    let message_path = private_temp_dir.join(format!("{stem}.message.bin"));
    let signature_path = private_temp_dir.join(format!("{stem}.signature.bin"));
    let mut der = hex_literal_ed25519_spki_prefix();
    der.extend_from_slice(public);
    let outcome = (|| {
        write_new(&public_path, &der)?;
        write_new(&message_path, message)?;
        write_new(&signature_path, signature)?;
        OpenOptions::new().read(true).open(private_temp_dir)
            .and_then(|file|file.sync_all())
            .map_err(|e|ReplayError::new(FAIL_SIGNATURE,format!("cannot sync verifier directory: {e}")))?;
        let output = Command::new(openssl_path)
            .args(["pkeyutl", "-verify", "-pubin", "-inkey"])
            .arg(&public_path)
            .args(["-rawin", "-in"])
            .arg(&message_path)
            .args(["-sigfile"])
            .arg(&signature_path)
            .env_clear()
            .env("LC_ALL","C")
            .env("LANG","C")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| ReplayError::new(FAIL_SIGNATURE, format!("cannot run OpenSSL verifier: {e}")))?;
        if !output.status.success() { return Err(ReplayError::new(FAIL_SIGNATURE, "Ed25519 verification failed")); }
        Ok(())
    })();
    let _ = fs::remove_file(public_path);
    let _ = fs::remove_file(message_path);
    let _ = fs::remove_file(signature_path);
    let _ = OpenOptions::new().read(true).open(private_temp_dir).and_then(|file|file.sync_all());
    outcome
}

fn hex_literal_ed25519_spki_prefix() -> Vec<u8> {
    vec![0x30,0x2a,0x30,0x05,0x06,0x03,0x2b,0x65,0x70,0x03,0x21,0x00]
}

fn typed_pair(universe: &[Vec<u8>], truth: &[Vec<u8>], signature: u64, input_tag: u64, input_schema: &[u8]) -> Result<(), ReplayError> {
    if universe.len() != truth.len() { return Err(ReplayError::new(FAIL_TYPED_BINDING, "typed row counts differ")); }
    for (index, (u_raw, t_raw)) in universe.iter().zip(truth).enumerate() {
        let u = decode_strict_cbor(u_raw).map_err(|e| ReplayError::new(FAIL_TYPED_BINDING, e.to_string()))?;
        let t = decode_strict_cbor(t_raw).map_err(|e| ReplayError::new(FAIL_TYPED_BINDING, e.to_string()))?;
        let ua = arr(&u, FAIL_TYPED_BINDING, "universe row")?;
        let ta = arr(&t, FAIL_TYPED_BINDING, "truth row")?;
        if uint(&ua[3], FAIL_TYPED_BINDING, "universe index")? != index as u64
            || uint(&ta[3], FAIL_TYPED_BINDING, "truth index")? != index as u64
            || uint(&ua[4], FAIL_TYPED_BINDING, "input signature")? != signature
        { return Err(ReplayError::new(FAIL_TYPED_BINDING, format!("typed row index/signature differs at {index}"))); }
        let nested = arr(&ua[5], FAIL_ROLE_BINDING, "canonical input")?;
        if nested.len() < 3 || nested[0] != CborValue::Unsigned(1) || nested[1] != CborValue::Unsigned(input_tag) || nested[2] != CborValue::Bytes(input_schema.to_vec()) {
            return Err(ReplayError::new(FAIL_ROLE_BINDING, format!("typed input has wrong role at {index}")));
        }
        let digest = content_hash(CANONICAL_INPUT_DOMAIN, &ua[5]).map_err(|e| ReplayError::new(FAIL_TYPED_BINDING, e.to_string()))?;
        if bytes(&ta[4], Some(32), FAIL_TYPED_BINDING, "truth input hash")? != digest {
            return Err(ReplayError::new(FAIL_TYPED_BINDING, format!("truth input hash differs at {index}")));
        }
    }
    Ok(())
}

fn verify_role_links(candidate: &[CborValue], decoded: &[Vec<CborValue>]) -> Result<(), ReplayError> {
    for (role_index, expected_role) in [(9usize,1u64),(10usize,2u64)] {
        let fields = arr(&decoded[role_index][0], FAIL_ROLE_BINDING, "role binding")?;
        if uint(&fields[3], FAIL_ROLE_BINDING, "target role")? != expected_role { return Err(ReplayError::new(FAIL_ROLE_BINDING, "role-binding role differs")); }
        for (source, target) in [(4,4),(5,5),(6,8),(7,9),(8,10),(9,11),(16,15),(17,16),(18,17)] {
            if fields[source] != candidate[target] { return Err(ReplayError::new(FAIL_ROLE_BINDING, "role-binding root splice")); }
        }
        let (universe, truth) = if expected_role == 1 { (24,25) } else { (26,27) };
        if fields[14] != candidate[universe] || fields[15] != candidate[truth] { return Err(ReplayError::new(FAIL_ROLE_BINDING, "role-binding typed-root splice")); }
    }
    let split = arr(&decoded[11][0], FAIL_SPLIT_BINDING, "split binding")?;
    for (source,target) in [(7,28),(8,29),(9,30),(10,31),(11,32),(12,33),(13,20),(14,21)] {
        if split[source] != candidate[target] { return Err(ReplayError::new(FAIL_SPLIT_BINDING, "split-binding root splice")); }
    }
    Ok(())
}

pub fn replay_package(package: &[u8], allow_authoritative: bool, private_temp_dir: &Path, openssl_path: &Path) -> Result<ReplayResult, ReplayError> {
    let package_value = decode_strict_cbor(package).map_err(|e| ReplayError::new(FAIL_PACKAGE_SCHEMA, e.to_string()))?;
    let p = arr(&package_value, FAIL_PACKAGE_SCHEMA, "package")?;
    if p.len() != 12 || p[0] != CborValue::Unsigned(1) || p[1] != CborValue::Unsigned(PACKAGE_TAG) || p[2] != CborValue::Bytes(PACKAGE_SCHEMA_ID.to_vec()) {
        return Err(ReplayError::new(FAIL_PACKAGE_SCHEMA, "package prefix/field count differs"));
    }
    let authoritative = match p[3] { CborValue::Bool(v) => v, _ => return Err(ReplayError::new(FAIL_PACKAGE_SCHEMA, "authority is not bool")) };
    if authoritative && !allow_authoritative { return Err(ReplayError::new(FAIL_PACKAGE_AUTHORITY, "authoritative replay requires runtime opt in")); }
    let purpose = uint(&p[4], FAIL_PURPOSE, "purpose")?;
    if !(1..=3).contains(&purpose) { return Err(ReplayError::new(FAIL_PURPOSE, "purpose must be 1, 2, or 3")); }
    let candidate_raw = bytes(&p[5], None, FAIL_CANDIDATE, "candidate")?;
    let bridge_raw = bytes(&p[6], None, FAIL_BRIDGE, "bridge")?;
    let nodes = arr(&p[7], FAIL_NODE_SET, "nodes")?;
    if nodes.len() != ROLES.len() { return Err(ReplayError::new(FAIL_NODE_SET, "node count differs")); }

    let candidate_value = decode_strict_cbor(candidate_raw).map_err(|e| ReplayError::new(FAIL_CANDIDATE, e.to_string()))?;
    let candidate = arr(&candidate_value, FAIL_CANDIDATE, "candidate")?;
    if candidate.len() != 47 || candidate[0] != CborValue::Unsigned(1) || candidate[1] != CborValue::Unsigned(0x310f) || candidate[2] != CborValue::Bytes(b"hegel-m3-execution-candidate/1".to_vec()) {
        return Err(ReplayError::new(FAIL_CANDIDATE, "candidate prefix/field count differs"));
    }
    if p[10] != candidate[45] || p[11] != candidate[46] { return Err(ReplayError::new(FAIL_CANDIDATE, "package time/commit differs")); }

    let mut roots = Vec::<[u8;32]>::new();
    let mut decoded_rows = Vec::<Vec<CborValue>>::new();
    let mut raw_rows = Vec::<Vec<Vec<u8>>>::new();
    for (spec, node_value) in ROLES.iter().zip(nodes) {
        let n = arr(node_value, FAIL_NODE_SCHEMA, "node")?;
        if n.len()!=8 || uint(&n[0],FAIL_NODE_SCHEMA,"role")? != spec.role || uint(&n[1],FAIL_NODE_SCHEMA,"op")? != spec.op || uint(&n[2],FAIL_NODE_SCHEMA,"tag")? != spec.tag || bytes(&n[3],None,FAIL_NODE_SCHEMA,"schema")? != spec.schema {
            return Err(ReplayError::new(FAIL_NODE_SCHEMA, format!("role {} schema differs",spec.role)));
        }
        match (spec.domain, &n[4]) {
            (Some(expected), CborValue::Bytes(v)) if v == expected.as_bytes() => {},
            (None, CborValue::Null) => {},
            _ => return Err(ReplayError::new(FAIL_NODE_SCHEMA, format!("role {} domain differs",spec.role))),
        }
        if uint(&n[6],FAIL_NODE_COUNT,"count")? as usize != spec.count { return Err(ReplayError::new(FAIL_NODE_COUNT, format!("role {} count differs",spec.role))); }
        let preimage_values = arr(&n[5],FAIL_NODE_PREIMAGE,"preimages")?;
        let raws: Vec<Vec<u8>> = preimage_values.iter().map(|v| bytes(v,None,FAIL_NODE_PREIMAGE,"preimage").map(|b|b.to_vec())).collect::<Result<_,_>>()?;
        let computed = if spec.op == 3 {
            if !raws.is_empty() { return Err(ReplayError::new(FAIL_NODE_PREIMAGE,"sealed rows were disclosed")); }
            root(&n[7],FAIL_NODE_PREIMAGE,"sealed root")?
        } else {
            if !matches!(n[7],CborValue::Null) || raws.len()!=spec.count { return Err(ReplayError::new(FAIL_NODE_COUNT,format!("role {} preimage count differs",spec.role))); }
            let values: Vec<CborValue> = raws.iter().map(|raw| decode_strict_cbor(raw).map_err(|e|ReplayError::new(FAIL_NODE_PREIMAGE,e.to_string()))).collect::<Result<_,_>>()?;
            for value in &values { prefix(value,spec)?; }
            if spec.op==1 { content_hash_cbor(spec.domain.expect("content domain"),&raws[0]).map_err(|e|ReplayError::new(FAIL_NODE_PREIMAGE,e.to_string()))? }
            else { rfc6962_canonical_record_root(&raws).map_err(|e|ReplayError::new(FAIL_NODE_PREIMAGE,e.to_string()))? }
        };
        if root(&candidate[spec.candidate_index],FAIL_ROOT_BINDING,"candidate role root")? != computed { return Err(ReplayError::new(FAIL_ROOT_BINDING,format!("candidate role {} differs",spec.role))); }
        let values = raws.iter().map(|raw|decode_strict_cbor(raw).map_err(|e|ReplayError::new(FAIL_NODE_PREIMAGE,e.to_string()))).collect::<Result<Vec<_>,_>>()?;
        roots.push(computed); decoded_rows.push(values); raw_rows.push(raws);
    }
    verify_role_links(candidate,&decoded_rows)?;
    typed_pair(&raw_rows[20],&raw_rows[21],1,0x3401,b"hegel-odd-input/1")?;
    typed_pair(&raw_rows[22],&raw_rows[23],2,0x3402,b"hegel-sink-input/1")?;

    let candidate_root = content_hash_cbor("HEGEL/M3_EXECUTION_CANDIDATE/V1",candidate_raw).map_err(|e|ReplayError::new(FAIL_CANDIDATE,e.to_string()))?;
    let bridge_value = decode_strict_cbor(bridge_raw).map_err(|e|ReplayError::new(FAIL_BRIDGE,e.to_string()))?;
    let bridge = arr(&bridge_value,FAIL_BRIDGE,"bridge")?;
    if bridge.len()!=10 || bridge[0]!=CborValue::Unsigned(1) || bridge[1]!=CborValue::Unsigned(0x310e) || bridge[2]!=CborValue::Bytes(b"hegel-bridge-replay-statement/1".to_vec())
        || bridge[3]!=candidate[3] || bridge[4]!=candidate[12] || bytes(&bridge[5],Some(32),FAIL_BRIDGE,"candidate root")? != candidate_root || bridge[6]!=candidate[4] || bridge[7]!=candidate[5] || bridge[8]!=candidate[23] || bridge[9]!=candidate[22]
    { return Err(ReplayError::new(FAIL_BRIDGE,"bridge does not exactly project candidate")); }
    let bridge_root = content_hash_cbor("HEGEL/BRIDGE_REPLAY_STATEMENT/V1",bridge_raw).map_err(|e|ReplayError::new(FAIL_BRIDGE,e.to_string()))?;

    let key_raw = bytes(&p[8],None,FAIL_TRUST_BINDING,"purpose1 key manifest")?;
    let key_value = decode_strict_cbor(key_raw).map_err(|e|ReplayError::new(FAIL_TRUST_BINDING,e.to_string()))?;
    let key = arr(&key_value,FAIL_TRUST_BINDING,"key manifest")?;
    if key.len()!=10 || key[0]!=CborValue::Unsigned(1) || key[1]!=CborValue::Unsigned(0x310c) || key[2]!=CborValue::Bytes(b"hegel-actor-key-manifest/1".to_vec()) || uint(&key[3],FAIL_TRUST_BINDING,"key purpose")? != 1 || key[9]!=candidate[46] {
        return Err(ReplayError::new(FAIL_TRUST_BINDING,"purpose1 key manifest differs"));
    }
    let key_root = content_hash_cbor("HEGEL/ACTOR_KEY_MANIFEST/V1",key_raw).map_err(|e|ReplayError::new(FAIL_TRUST_BINDING,e.to_string()))?;
    let trust = arr(&decoded_rows[19][0],FAIL_TRUST_BINDING,"actor trust")?;
    let entries = arr(&trust[4],FAIL_TRUST_BINDING,"purpose entries")?;
    if entries.len()!=4 { return Err(ReplayError::new(FAIL_TRUST_BINDING,"actor trust purpose count differs")); }
    for (i,entry) in entries.iter().enumerate() {
        let e=arr(entry,FAIL_TRUST_BINDING,"purpose entry")?;
        if e.len()!=2 || uint(&e[0],FAIL_TRUST_BINDING,"entry purpose")? != (i+1) as u64 { return Err(ReplayError::new(FAIL_TRUST_BINDING,"actor trust purpose set differs")); }
    }
    if bytes(&arr(&entries[0],FAIL_TRUST_BINDING,"purpose1 entry")?[1],Some(32),FAIL_TRUST_BINDING,"key root")? != key_root { return Err(ReplayError::new(FAIL_TRUST_BINDING,"purpose1 key is not trust-bound")); }
    let key_id=bytes(&key[4],Some(16),FAIL_TRUST_BINDING,"key id")?;
    let public=bytes(&key[5],Some(32),FAIL_TRUST_BINDING,"public key")?;
    let digest:[u8;32]=Sha256::digest(public).into();
    if key_id != &digest[..16] { return Err(ReplayError::new(FAIL_TRUST_BINDING,"key ID derivation differs")); }
    let created=uint(&candidate[45],FAIL_TRUST_BINDING,"created")?;
    let valid_from=uint(&key[7],FAIL_TRUST_BINDING,"valid from")?;
    let valid_until=match key[8] { CborValue::Null=>None,CborValue::Unsigned(v)=>Some(v),_=>return Err(ReplayError::new(FAIL_TRUST_BINDING,"valid until type differs")) };
    if valid_from>created || valid_until.is_some_and(|v|created>v) { return Err(ReplayError::new(FAIL_TRUST_BINDING,"purpose1 key is outside validity")); }
    let verified = if purpose==1 {
        if !matches!(p[9],CborValue::Null) { return Err(ReplayError::new(FAIL_SIGNATURE_PHASE,"purpose1 package must be unsigned")); }
        false
    } else {
        let signature=bytes(&p[9],Some(64),FAIL_SIGNATURE,"signature")?;
        verify_ed25519_openssl(
            public,
            signature,
            &bridge_attestation_signature_preimage_v1(
                &bridge_root,
                1,
                uint(&key[6], FAIL_TRUST_BINDING, "key epoch")?,
            )?,
            private_temp_dir,
            openssl_path,
        )?;
        true
    };
    let package_digest=content_hash(PACKAGE_HASH_DOMAIN,&package_value).map_err(|e|ReplayError::new(FAIL_PACKAGE_SCHEMA,e.to_string()))?;
    Ok(ReplayResult{package_digest,candidate_root,bridge_root,purpose,purpose1_signature_verified:verified,authoritative})
}

pub fn replay_file(path: &Path, allow_authoritative: bool, private_temp_dir: &Path, openssl_path: &Path) -> Result<ReplayResult, ReplayError> {
    let payload=fs::read(path).map_err(|e|ReplayError::new(FAIL_PACKAGE_SCHEMA,format!("cannot read package: {e}")))?;
    replay_package(&payload,allow_authoritative,private_temp_dir,openssl_path)
}
