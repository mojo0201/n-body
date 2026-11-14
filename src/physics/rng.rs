//! Seed encoding (base62) and the splitmix32 RNG core.
//!
//! Seeds are 16 base62 characters, big-endian and zero-padded, giving
//! `62^16 ≈ 4.77e28` distinct values — just under 2^96, so a `u128` holds
//! the whole domain. The RNG state itself stays 32 bits wide: widening it
//! would change the output stream, and stream stability is a hard contract
//! (seeds below 2^32 must keep reproducing their historical sequences).

/// Base62 alphabet used for both formatting and parsing seeds.
/// Order is significant: index == digit value. `0-9 < a-z < A-Z`.
pub const SEED_ALPHABET: &[u8; 62] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Width of a formatted seed in base62 characters.
/// 62^16 < 2^96, so every value fits a `u128` with 32 bits to spare.
pub const SEED_LEN: usize = 16;

/// Map an ASCII byte to its base62 digit value (0..62).
///
/// Explicit range checks beat scanning `SEED_ALPHABET`: three predictable
/// comparisons per byte instead of up to 62, and the branches are nearly
/// always taken the same way for a given input class.
#[inline]
fn decode_byte(c: u8) -> Option<u64> {
    match c {
        b'0'..=b'9' => Some((c - b'0') as u64),
        b'a'..=b'z' => Some((c - b'a') as u64 + 10),
        b'A'..=b'Z' => Some((c - b'A') as u64 + 36),
        _ => None,
    }
}

/// Largest value expressible in `SEED_LEN` base62 chars.
/// Formatting clamps to this so a caller that hands you any `u128`
/// still gets a round-trippable string.
pub const SEED_SPACE: u128 = 62u128.pow(SEED_LEN as u32); // 62^16

#[inline]
pub fn seed_to_base62(mut seed: u128) -> String {
    seed %= SEED_SPACE; // <-- new; u128 has ~32 bits of headroom above 62^16
    let mut buf = [0u8; SEED_LEN];
    for k in (0..SEED_LEN).rev() {
        buf[k] = SEED_ALPHABET[(seed % 62) as usize];
        seed /= 62;
    }
    unsafe { String::from_utf8_unchecked(buf.into()) }
}

/// Parse a base62 seed of 1 to `SEED_LEN` characters.
///
/// Returns `None` for empty input, over-long input, or any non-base62 byte.
/// Leading zeros are allowed. The accumulator cannot overflow: `len <= 16`
/// bounds the running value below `62^16 < 2^96`, so plain `u128` arithmetic
/// is used instead of checked ops (two branches saved per character).
#[inline]
pub fn seed_from_base62(s: &str) -> Option<u128> {
    let bytes = s.as_bytes();
    if bytes.is_empty() || bytes.len() > SEED_LEN {
        return None;
    }
    let mut v: u128 = 0;
    for &c in bytes {
        v = v * 62 + decode_byte(c)? as u128;
    }
    Some(v)
}

/// 32-bit mulberry32 generator.
///
/// The state is a single word. All RNG output is derived from it via the
/// constants below; do not touch `next_u32` — its exact bit sequence is part
/// of the reproducibility contract.
pub struct XorShift {
    pub(super) s: u32,
}

impl XorShift {
    /// Fold a 128-bit seed into the 32-bit RNG state.
    ///
    /// The four 32-bit lanes are XOR-combined, then clamped to be non-zero
    /// (a zero state stalls the generator). Consequences worth knowing:
    ///
    /// * Backward compatible: for `seed < 2^32` the upper three lanes are
    ///   zero, so the fold is a no-op and the resulting stream is identical
    ///   to the pre-16-char versions of this crate.
    /// * Entropy-limited: only 32 bits of the 96-bit seed actually affect
    ///   the stream; seeds that agree in their lane-wise XOR collide. Using
    ///   the full seed would require a wider state, which would break the
    ///   stream-stability contract above.
    #[inline]
    pub fn new(seed: u128) -> Self {
        let s =
            (seed as u32) ^ ((seed >> 32) as u32) ^ ((seed >> 64) as u32) ^ ((seed >> 96) as u32);
        Self { s: s.max(1) }
    }

    /// Advance the state and emit the next 32-bit word.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.s = self.s.wrapping_add(0x6d2b79f5);
        let mut t = self.s;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        t ^ (t >> 14)
    }

    /// Uniform float in `[0, 1)`.
    ///
    /// Multiplication by the exact constant `2^-32` replaces the division:
    /// both are powers of two and both are exact in `f64`, so the result is
    /// bit-identical to `x / 2^32` for every `u32`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) * (1.0 / 4_294_967_296.0)
    }
}
