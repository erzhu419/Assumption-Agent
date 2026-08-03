#!/bin/sh
# Purpose-3-only M2.5 actor.  It is deliberately not a generic signing oracle:
# the committed Rust formal replayer must strictly decode and ContentHash the
# bridge statement before the purpose-3 key can be used.

set -eu
umask 077
# POSIX shells synthesize PWD even under `env -i`; remove it so the launch has
# exactly 19 keys.  The raw path is then removed from the worker environment,
# leaving the 18-key safe allowlist for every non-probe descendant.
unset PWD

fail() {
    printf '%s\n' "$1" >&2
    exit 70
}

require_hex_length() {
    value=$1
    length=$2
    [ "${#value}" -eq "$length" ] || fail FAIL_M25_RUST_ACTOR_ENVIRONMENT
    case "$value" in
        *[!0-9a-f]*) fail FAIL_M25_RUST_ACTOR_ENVIRONMENT ;;
    esac
}

[ -n "${HEGEL_HOST_REPOSITORY_PATH:-}" ] \
    || fail FAIL_M25_RUST_ACTOR_HOST_REPOSITORY_PATH
[ "$(/usr/bin/env | /usr/bin/wc -l)" -eq 19 ] \
    || fail FAIL_M25_RUST_ACTOR_ENVIRONMENT_CARDINALITY
# Preserve the value only as a non-exported shell variable, then remove it
# from the real worker environment before any replay, OpenSSL, key, or seed
# child can inherit it.  It is injected once into the dedicated live probe.
host_repository_path=$HEGEL_HOST_REPOSITORY_PATH
unset HEGEL_HOST_REPOSITORY_PATH

[ "${HEGEL_ACTOR_PROFILE_ID:-}" = "hegel-owner-accepted-container-technical-actors-v1" ] \
    || fail FAIL_M25_RUST_ACTOR_PROFILE
[ "${HEGEL_PURPOSE_ID:-}" = "3" ] || fail FAIL_M25_RUST_ACTOR_PURPOSE
[ "$(/usr/bin/id -u)" = "65534" ] && [ "$(/usr/bin/id -g)" = "65534" ] \
    || fail FAIL_M25_RUST_ACTOR_IDENTITY
[ "$#" -eq 1 ] || fail FAIL_M25_RUST_ACTOR_ARGUMENTS
[ "${HEGEL_OPERATION_ID:-}" = "$1" ] || fail FAIL_M25_RUST_ACTOR_OPERATION_BINDING
[ "${HEGEL_PROBE_INPUT_WRITE_PATH:-}" = "/input/.hegel-write-probe" ] \
    || fail FAIL_M25_RUST_ACTOR_PROBE_PATH
[ "${LANG:-}" = C ] && [ "${LC_ALL:-}" = C.UTF-8 ] \
    && [ "${PATH:-}" = /usr/local/bin:/usr/bin:/bin ] \
    && [ "${PYTHONDONTWRITEBYTECODE:-}" = 1 ] \
    && [ "${PYTHONHASHSEED:-}" = 0 ] \
    || fail FAIL_M25_RUST_ACTOR_ENVIRONMENT
require_hex_length "${HEGEL_BASIS_COMMIT:-}" 40
require_hex_length "${HEGEL_DAEMON_RECEIPT_SHA256:-}" 64
require_hex_length "${HEGEL_PROFILE_SHA256:-}" 64
require_hex_length "${HEGEL_HOST_REPOSITORY_PATH_SHA256:-}" 64
host_repository_path_sha256=$(
    /usr/bin/printf %s "$host_repository_path" | /usr/bin/sha256sum
) || fail FAIL_M25_RUST_ACTOR_HOST_REPOSITORY_PATH_HASH
host_repository_path_sha256=${host_repository_path_sha256%% *}
[ "$host_repository_path_sha256" = "$HEGEL_HOST_REPOSITORY_PATH_SHA256" ] \
    || fail FAIL_M25_RUST_ACTOR_HOST_REPOSITORY_PATH_HASH
require_hex_length "${HEGEL_RUN_ID:-}" 32
require_hex_length "${HEGEL_OPERATION_NONCE:-}" 32
require_hex_length "${HEGEL_OPERATION_REQUEST_SHA256:-}" 64
case "${HEGEL_OPERATION_SEQUENCE:-}" in
    ''|0|*[!0-9]*) fail FAIL_M25_RUST_ACTOR_OPERATION_SEQUENCE ;;
esac
case "${HEGEL_ACTOR_IMAGE_REF:-}" in
    *@sha256:*) ;;
    *) fail FAIL_M25_RUST_ACTOR_IMAGE_BINDING ;;
esac
require_hex_length "${HEGEL_ACTOR_IMAGE_REF##*@sha256:}" 64
case "$1" in
    qualify-only|keygen|keygen-resume|bridge-replay-sign-rust) ;;
    *) fail FAIL_M25_RUST_ACTOR_OPERATION ;;
esac

probe_receipt="/output/operation-probe-$1.json"
probe_temporary="/output/.operation-probe-$1.tmp"
rust_probe_receipt="/output/operation-rust-probe-$1.json"
rust_probe_temporary="/output/.operation-rust-probe-$1.tmp"
[ ! -e "$probe_receipt" ] && [ ! -e "$probe_temporary" ] \
    && [ ! -e "$rust_probe_receipt" ] && [ ! -e "$rust_probe_temporary" ] \
    || fail FAIL_M25_RUST_ACTOR_PROBE_REPLAY
: > /state/.hegel-operation-write-probe \
    && /usr/bin/rm /state/.hegel-operation-write-probe \
    || fail FAIL_M25_RUST_ACTOR_STATE_PROBE
HEGEL_HOST_REPOSITORY_PATH=$host_repository_path \
    /input/rust-live-probe >"$rust_probe_temporary" 2>/dev/null \
    || fail FAIL_M25_RUST_ACTOR_LIVE_PROBE
unset host_repository_path
[ "$(/usr/bin/wc -l <"$rust_probe_temporary")" -eq 1 ] \
    || fail FAIL_M25_RUST_ACTOR_LIVE_PROBE_FRAMING
rust_probe_sha256=$(/usr/bin/sha256sum "$rust_probe_temporary") \
    || fail FAIL_M25_RUST_ACTOR_LIVE_PROBE_HASH
rust_probe_sha256=${rust_probe_sha256%% *}
require_hex_length "$rust_probe_sha256" 64
/usr/bin/chmod 0644 "$rust_probe_temporary" \
    || fail FAIL_M25_RUST_ACTOR_LIVE_PROBE_MODE
/usr/bin/mv "$rust_probe_temporary" "$rust_probe_receipt" \
    || fail FAIL_M25_RUST_ACTOR_LIVE_PROBE_COMMIT
printf '{"operation_id":"%s","operation_nonce_hex":"%s","operation_request_sha256":"%s","operation_sequence":%s,"parent_environment_count":19,"parent_pid":%s,"purpose_id":3,"rust_probe_sha256":"%s","schema":"hegel-phase3-m25-rust-operation-parent-binding/1"}\n' \
    "$1" "$HEGEL_OPERATION_NONCE" "$HEGEL_OPERATION_REQUEST_SHA256" \
    "$HEGEL_OPERATION_SEQUENCE" "$$" "$rust_probe_sha256" \
    >"$probe_temporary" || fail FAIL_M25_RUST_ACTOR_PARENT_RECEIPT
/usr/bin/chmod 0644 "$probe_temporary" || fail FAIL_M25_RUST_ACTOR_PARENT_MODE
/usr/bin/mv "$probe_temporary" "$probe_receipt" \
    || fail FAIL_M25_RUST_ACTOR_PARENT_COMMIT

[ "$1" != "qualify-only" ] || exit 0

private_key=/state/ed25519-private.pem
public_der=/output/ed25519-public.der
signature=/output/ed25519-signature.bin

case "$1" in
    keygen|keygen-resume)
        key_operation=$1
        [ ! -e "$public_der" ] || fail FAIL_M25_RUST_ACTOR_PUBLIC_KEY_ALREADY_EXISTS
        /usr/bin/chmod 0700 /state || fail FAIL_M25_RUST_ACTOR_STATE_MODE
        if [ -e "$private_key" ]; then
            [ -f "$private_key" ] && [ ! -L "$private_key" ] \
                && [ "$(/usr/bin/stat -c %a "$private_key")" = 600 ] \
                || fail FAIL_M25_RUST_ACTOR_KEY_RECOVERY_STATE
        else
            [ "$key_operation" = keygen ] \
                || fail FAIL_M25_RUST_ACTOR_RECOVERY_KEY_ABSENT
            /usr/bin/openssl genpkey -algorithm ED25519 -out "$private_key" \
                >/dev/null 2>/dev/null || fail FAIL_M25_RUST_ACTOR_KEYGEN
            /usr/bin/chmod 0600 "$private_key" || fail FAIL_M25_RUST_ACTOR_KEY_MODE
        fi
        /usr/bin/openssl pkey -in "$private_key" -pubout -outform DER -out "$public_der" \
            >/dev/null 2>/dev/null || fail FAIL_M25_RUST_ACTOR_PUBLIC_KEY
        [ "$(/usr/bin/wc -c <"$public_der")" -eq 44 ] \
            || fail FAIL_M25_RUST_ACTOR_PUBLIC_KEY_SIZE
        /usr/bin/chmod 0644 "$public_der" || fail FAIL_M25_RUST_ACTOR_PUBLIC_MODE
        ;;
    bridge-replay-sign-rust)
        [ -f "$private_key" ] && [ ! -e "$signature" ] \
            || fail FAIL_M25_RUST_ACTOR_KEY_STATE
        for file in \
            /input/rust-bridge-dag-replay \
            /input/bridge-dag-package.cbor
        do
            [ -f "$file" ] || fail FAIL_M25_RUST_ACTOR_REPLAY_INPUT
        done
        [ -x /input/rust-bridge-dag-replay ] \
            || fail FAIL_M25_RUST_ACTOR_REPLAY_EXECUTABLE
        verifier_dir="/tmp/hegel-m25-bridge-verifier-p3-$HEGEL_OPERATION_NONCE"
        replay_temporary="$verifier_dir/replay-receipt.json"
        signing_preimage="$verifier_dir/signing-preimage.bin"
        replay_receipt=/output/bridge-dag-replay-receipt.json
        [ ! -e "$verifier_dir" ] && [ ! -e "$replay_receipt" ] \
            || fail FAIL_M25_RUST_ACTOR_REPLAY_STATE
        /usr/bin/mkdir -m 0700 "$verifier_dir" \
            || fail FAIL_M25_RUST_ACTOR_REPLAY_PRIVATE_TEMP
        cleanup_bridge_replay() {
            /usr/bin/rm -f "$signing_preimage" "$replay_temporary"
            /usr/bin/rmdir "$verifier_dir" 2>/dev/null || true
        }
        trap cleanup_bridge_replay EXIT HUP INT TERM
        /input/rust-bridge-dag-replay \
            --authoritative-runtime \
            --expected-purpose 3 \
            --signature-preimage-out "$signing_preimage" \
            /input/bridge-dag-package.cbor "$verifier_dir" \
            >"$replay_temporary" 2>/dev/null \
            || fail FAIL_M25_RUST_ACTOR_FULL_DAG_REPLAY
        [ -f "$signing_preimage" ] \
            && [ ! -L "$signing_preimage" ] \
            && [ "$(/usr/bin/stat -c %a "$signing_preimage")" = 600 ] \
            && [ "$(/usr/bin/wc -c <"$signing_preimage")" -eq 80 ] \
            || fail FAIL_M25_RUST_ACTOR_PREIMAGE_STATE
        [ "$(/usr/bin/wc -l <"$replay_temporary")" -eq 1 ] \
            && /usr/bin/grep -q '"authoritative":true' "$replay_temporary" \
            && /usr/bin/grep -q '"purpose":3' "$replay_temporary" \
            && /usr/bin/grep -q '"purpose1_signature_verified":true' "$replay_temporary" \
            && /usr/bin/grep -q '"status":"PASS"' "$replay_temporary" \
            || fail FAIL_M25_RUST_ACTOR_REPLAY_RECEIPT
        /usr/bin/openssl pkeyutl -sign -rawin -inkey "$private_key" \
            -in "$signing_preimage" -out "$signature" \
            >/dev/null 2>/dev/null || fail FAIL_M25_RUST_ACTOR_SIGNATURE
        [ "$(/usr/bin/wc -c <"$signature")" -eq 64 ] \
            || fail FAIL_M25_RUST_ACTOR_SIGNATURE_SIZE
        /usr/bin/chmod 0644 "$signature" || fail FAIL_M25_RUST_ACTOR_SIGNATURE_MODE
        /usr/bin/chmod 0644 "$replay_temporary" \
            || fail FAIL_M25_RUST_ACTOR_REPLAY_RECEIPT_MODE
        /usr/bin/mv "$replay_temporary" "$replay_receipt" \
            || fail FAIL_M25_RUST_ACTOR_REPLAY_RECEIPT_COMMIT
        /usr/bin/rm -f "$signing_preimage" \
            || fail FAIL_M25_RUST_ACTOR_PREIMAGE_CLEANUP
        /usr/bin/rmdir "$verifier_dir" \
            || fail FAIL_M25_RUST_ACTOR_PRIVATE_TEMP_CLEANUP
        trap - EXIT HUP INT TERM
        ;;
    *)
        fail FAIL_M25_RUST_ACTOR_OPERATION
        ;;
esac

exit 0
