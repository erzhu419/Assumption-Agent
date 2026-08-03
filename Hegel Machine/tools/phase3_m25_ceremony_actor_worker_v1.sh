#!/bin/sh
# Public-input/private-volume Ed25519 worker for M2.5 purpose containers.
#
# The only argument is an allowlisted public operation.  Key material stays in
# /state; sign/verify messages arrive as public files in /input; public outputs
# are written to /output.  stdin, environment, and argv never carry secrets.

set -eu
umask 077

fail() {
    # Stable code only.  Never print an OpenSSL diagnostic that might contain
    # a secret-state path or secret-bearing value.
    printf '%s\n' "$1" >&2
    exit 70
}

[ "$#" -eq 1 ] || fail FAIL_M25_ACTOR_WORKER_ARGUMENTS
[ "${HEGEL_ACTOR_PROFILE_ID:-}" = "hegel-owner-accepted-container-technical-actors-v1" ] \
    || fail FAIL_M25_ACTOR_WORKER_PROFILE

case "${HEGEL_PURPOSE_ID:-}" in
    1|2|3|4) ;;
    *) fail FAIL_M25_ACTOR_WORKER_PURPOSE ;;
esac

PRIVATE_KEY=/state/ed25519-private.pem
PUBLIC_DER=/output/ed25519-public.der
SIGNING_INPUT=/input/signing-preimage.bin
SIGNATURE_OUTPUT=/output/ed25519-signature.bin

case "$1" in
    keygen)
        [ ! -e "$PRIVATE_KEY" ] || fail FAIL_M25_ACTOR_KEY_ALREADY_EXISTS
        [ ! -e "$PUBLIC_DER" ] || fail FAIL_M25_ACTOR_PUBLIC_OUTPUT_EXISTS
        openssl genpkey -algorithm ED25519 -out "$PRIVATE_KEY" \
            >/dev/null 2>/dev/null || fail FAIL_M25_ACTOR_KEYGEN
        chmod 0600 "$PRIVATE_KEY" || fail FAIL_M25_ACTOR_KEY_PERMISSIONS
        openssl pkey -in "$PRIVATE_KEY" -pubout -outform DER -out "$PUBLIC_DER" \
            >/dev/null 2>/dev/null || fail FAIL_M25_ACTOR_PUBLIC_KEY_EXPORT
        chmod 0644 "$PUBLIC_DER" || fail FAIL_M25_ACTOR_PUBLIC_KEY_PERMISSIONS
        ;;
    sign)
        [ -f "$PRIVATE_KEY" ] || fail FAIL_M25_ACTOR_KEY_MISSING
        [ -f "$SIGNING_INPUT" ] || fail FAIL_M25_ACTOR_SIGNING_INPUT_MISSING
        [ ! -e "$SIGNATURE_OUTPUT" ] || fail FAIL_M25_ACTOR_SIGNATURE_EXISTS
        [ "$(wc -c <"$SIGNING_INPUT")" -le 4096 ] \
            || fail FAIL_M25_ACTOR_SIGNING_INPUT_TOO_LARGE
        openssl pkeyutl -sign -rawin -inkey "$PRIVATE_KEY" \
            -in "$SIGNING_INPUT" -out "$SIGNATURE_OUTPUT" \
            >/dev/null 2>/dev/null || fail FAIL_M25_ACTOR_SIGNATURE
        [ "$(wc -c <"$SIGNATURE_OUTPUT")" -eq 64 ] \
            || fail FAIL_M25_ACTOR_SIGNATURE_SIZE
        chmod 0644 "$SIGNATURE_OUTPUT" || fail FAIL_M25_ACTOR_SIGNATURE_PERMISSIONS
        ;;
    verify)
        [ -f /input/ed25519-public.pem ] || fail FAIL_M25_ACTOR_PUBLIC_KEY_MISSING
        [ -f "$SIGNING_INPUT" ] || fail FAIL_M25_ACTOR_SIGNING_INPUT_MISSING
        [ -f /input/ed25519-signature.bin ] || fail FAIL_M25_ACTOR_SIGNATURE_MISSING
        openssl pkeyutl -verify -rawin -pubin -inkey /input/ed25519-public.pem \
            -in "$SIGNING_INPUT" -sigfile /input/ed25519-signature.bin \
            >/dev/null 2>/dev/null || fail FAIL_M25_ACTOR_SIGNATURE_INVALID
        ;;
    *)
        fail FAIL_M25_ACTOR_WORKER_OPERATION
        ;;
esac

exit 0
