#!/bin/sh
# Purpose-separated, qualification-only finalizer. Never accepts formal input,
# a seed operation, an arbitrary output path, or a caller-selected key.
set -eu
umask 077
unset PWD

fail() { printf '%s\n' "$1" >&2; exit 70; }
hexlen() {
    [ "${#1}" -eq "$2" ] || fail FAIL_M25_QUALIFICATION_FINALIZE_ENV
    case "$1" in *[!0-9a-f]*) fail FAIL_M25_QUALIFICATION_FINALIZE_ENV ;; esac
}

[ "$#" -eq 1 ] && [ "$1" = qualification-finalize ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_OPERATION
[ "${HEGEL_OPERATION_ID:-}" = qualification-finalize ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_OPERATION
case "${HEGEL_PURPOSE_ID:-}" in 1|2|3|4) ;; *) fail FAIL_M25_QUALIFICATION_FINALIZE_PURPOSE ;; esac
[ -n "${HEGEL_HOST_REPOSITORY_PATH:-}" ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_HOST_REPOSITORY_PATH
[ "$(/usr/bin/env | /usr/bin/wc -l)" -eq 20 ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_ENV
# Keep the raw path out of every validation and cryptographic descendant.  A
# non-exported copy is passed only to the purpose-specific live probe below.
host_repository_path=$HEGEL_HOST_REPOSITORY_PATH
unset HEGEL_HOST_REPOSITORY_PATH
[ "$(/usr/bin/id -u)" = 65534 ] && [ "$(/usr/bin/id -g)" = 65534 ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_IDENTITY
hexlen "${HEGEL_BASIS_COMMIT:-}" 40
hexlen "${HEGEL_DAEMON_RECEIPT_SHA256:-}" 64
hexlen "${HEGEL_HOST_REPOSITORY_PATH_SHA256:-}" 64
hexlen "${HEGEL_PROFILE_SHA256:-}" 64
hexlen "${HEGEL_RUN_ID:-}" 32
hexlen "${HEGEL_OPERATION_NONCE:-}" 32
hexlen "${HEGEL_OPERATION_REQUEST_SHA256:-}" 64
hexlen "${HEGEL_QUALIFICATION_PREIMAGE_SHA256:-}" 64
case "${HEGEL_OPERATION_SEQUENCE:-}" in ''|0|*[!0-9]*) fail FAIL_M25_QUALIFICATION_FINALIZE_SEQUENCE ;; esac
case "${HEGEL_ACTOR_IMAGE_REF:-}" in *@sha256:*) ;; *) fail FAIL_M25_QUALIFICATION_FINALIZE_IMAGE ;; esac
hexlen "${HEGEL_ACTOR_IMAGE_REF##*@sha256:}" 64
[ "${HEGEL_ACTOR_PROFILE_ID:-}" = hegel-owner-accepted-container-technical-actors-v1 ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_PROFILE
[ "${PATH:-}" = /usr/local/bin:/usr/bin:/bin ] \
    && [ "${LANG:-}" = C ] && [ "${LC_ALL:-}" = C.UTF-8 ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_ENV
host_repository_path_sha256=$(
    /usr/bin/printf %s "$host_repository_path" | /usr/bin/sha256sum
) || fail FAIL_M25_QUALIFICATION_FINALIZE_HOST_REPOSITORY_PATH
host_repository_path_sha256=${host_repository_path_sha256%% *}
[ "$host_repository_path_sha256" = "$HEGEL_HOST_REPOSITORY_PATH_SHA256" ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_HOST_REPOSITORY_PATH

request=/input/qualification-finalize-request.json
statement=/input/qualification-finalize-statement.json
preimage=/input/qualification-finalize-preimage.bin
private_key=/state/ed25519-private.pem
signature=/output/qualification-finalize-signature.bin
receipt=/output/qualification-finalize-probe.json
probe=/output/qualification-finalize-live-probe.json
for path in "$request" "$statement" "$preimage" "$private_key"; do
    [ -f "$path" ] && [ ! -L "$path" ] || fail FAIL_M25_QUALIFICATION_FINALIZE_INPUT
done
[ ! -e "$signature" ] && [ ! -e "$receipt" ] && [ ! -e "$probe" ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_REPLAY

request_sha=$(/usr/bin/sha256sum "$request") || fail FAIL_M25_QUALIFICATION_FINALIZE_REQUEST
request_sha=${request_sha%% *}
[ "$request_sha" = "$HEGEL_OPERATION_REQUEST_SHA256" ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_REQUEST
preimage_sha=$(/usr/bin/sha256sum "$preimage") || fail FAIL_M25_QUALIFICATION_FINALIZE_PREIMAGE
preimage_sha=${preimage_sha%% *}
[ "$preimage_sha" = "$HEGEL_QUALIFICATION_PREIMAGE_SHA256" ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_PREIMAGE

# Independently reconstruct both frozen cryptographic preimages.  The
# supervisor cannot substitute arbitrary bytes while retaining a valid worker
# receipt: the statement digest, purpose octet, domain separators and complete
# canonical request all have to agree inside this key-bearing actor.
[ "$(/usr/bin/wc -l <"$statement")" -eq 1 ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_STATEMENT
case "$HEGEL_PURPOSE_ID" in
    1) purpose_oct='\001' ;;
    2) purpose_oct='\002' ;;
    3) purpose_oct='\003' ;;
    4) purpose_oct='\004' ;;
esac
expected_preimage=/tmp/qualification-finalize-expected-preimage.bin
statement_hash_input=/tmp/qualification-finalize-statement-hash-input.bin
{
    printf 'HEGEL/M25/PROTOCOL_QUALIFICATION_STATEMENT/V1\000'
    /usr/bin/cat "$statement"
} >"$statement_hash_input" || fail FAIL_M25_QUALIFICATION_FINALIZE_STATEMENT
{
    printf 'HEGEL/M25/PROTOCOL_QUALIFICATION_FINALIZE_SIGNATURE/V1\000'
    printf '%b' "$purpose_oct"
    /usr/bin/openssl dgst -sha256 -binary "$statement_hash_input"
} >"$expected_preimage" || fail FAIL_M25_QUALIFICATION_FINALIZE_PREIMAGE
/usr/bin/cmp -s "$expected_preimage" "$preimage" \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_PREIMAGE
statement_text=$(/usr/bin/cat "$statement") \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_STATEMENT
expected_request=/tmp/qualification-finalize-expected-request.json
printf '{"preimage_sha256":"sha256:%s","purpose_id":%s,"schema":"hegel-phase3-m25-protocol-qualification-finalize-request/1","statement":%s}\n' \
    "$preimage_sha" "$HEGEL_PURPOSE_ID" "$statement_text" >"$expected_request" \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_REQUEST
/usr/bin/cmp -s "$expected_request" "$request" \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_REQUEST

if [ -x /input/rust-live-probe ]; then
    HEGEL_HOST_REPOSITORY_PATH=$host_repository_path \
        /input/rust-live-probe >"$probe" 2>/dev/null \
        || fail FAIL_M25_QUALIFICATION_FINALIZE_LIVE_PROBE
else
    HEGEL_HOST_REPOSITORY_PATH=$host_repository_path \
        /usr/local/bin/python3 -I -B /input/tools/phase3_container_actor_probe_v1.py \
        >"$probe" 2>/dev/null || fail FAIL_M25_QUALIFICATION_FINALIZE_LIVE_PROBE
fi
unset host_repository_path
[ "$(/usr/bin/wc -l <"$probe")" -eq 1 ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_LIVE_PROBE
probe_sha=$(/usr/bin/sha256sum "$probe") || fail FAIL_M25_QUALIFICATION_FINALIZE_LIVE_PROBE
probe_sha=${probe_sha%% *}

/usr/bin/openssl pkeyutl -sign -rawin -inkey "$private_key" \
    -in "$preimage" -out "$signature" >/dev/null 2>/dev/null \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_SIGNATURE
[ "$(/usr/bin/wc -c <"$signature")" -eq 64 ] \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_SIGNATURE
/usr/bin/chmod 0644 "$signature" "$probe"
signature_sha=$(/usr/bin/sha256sum "$signature") || fail FAIL_M25_QUALIFICATION_FINALIZE_SIGNATURE
signature_sha=${signature_sha%% *}

temporary=/output/.qualification-finalize-probe.tmp
printf '{"live_probe_sha256":"%s","operation_id":"qualification-finalize","operation_nonce_hex":"%s","operation_request_sha256":"%s","operation_sequence":%s,"preimage_sha256":"%s","purpose_id":%s,"schema":"hegel-phase3-m25-protocol-qualification-finalize-probe/1","signature_sha256":"%s"}\n' \
    "$probe_sha" "$HEGEL_OPERATION_NONCE" "$HEGEL_OPERATION_REQUEST_SHA256" \
    "$HEGEL_OPERATION_SEQUENCE" "$preimage_sha" "$HEGEL_PURPOSE_ID" \
    "$signature_sha" >"$temporary" \
    || fail FAIL_M25_QUALIFICATION_FINALIZE_RECEIPT
/usr/bin/chmod 0644 "$temporary"
/usr/bin/mv "$temporary" "$receipt" || fail FAIL_M25_QUALIFICATION_FINALIZE_RECEIPT
exit 0
