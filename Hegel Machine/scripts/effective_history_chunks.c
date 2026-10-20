/* Stable content boundaries let insertions reuse the unchanged snapshot blocks. */
#include <stddef.h>
#include <stdint.h>

static uint64_t gear[256];

void history_chunks_init(void) {
    uint64_t x = 0x7d6e4b910abc235fULL;
    for (int i = 0; i < 256; ++i) {
        x += 0x9e3779b97f4a7c15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        gear[i] = z ^ (z >> 31);
    }
}

size_t history_chunk_end(const unsigned char *data, size_t length) {
    const size_t minimum = 1024, maximum = 16384;
    uint64_t state = 0;
    size_t limit = length < maximum ? length : maximum;
    for (size_t i = minimum; i < limit; ++i) {
        state = (state << 1) + gear[data[i]];
        if ((state & 4095) == 0) return i + 1;
    }
    return length >= maximum ? maximum : 0;
}
