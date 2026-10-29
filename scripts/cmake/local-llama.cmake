# Meetily uses llama-helper only for local inference. Its bundled HTTP client is
# unused; disabling TLS also avoids the vendored client's OpenSSL 4 API mismatch.
set(LLAMA_OPENSSL OFF CACHE BOOL "Local inference needs no HTTP TLS client" FORCE)
