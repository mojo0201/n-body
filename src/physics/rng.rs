//! Seed alphabet (base62) and the xorshift RNG.

/// Base62 alphabet for seed display and input.
pub const SEED_ALPHABET: &[u8; 62] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Seed as 6 base62 characters, big-endian, zero-padded. 62^6 > 2^32.
pub fn seed_to_base62(mut seed: u32) -> String {
    let mut buf = [b'0'; 6];
    for k in (0..6).rev() {
        buf[k] = SEED_ALPHABET[(seed as usize) % 62];
        seed /= 62;
    }
    String::from_utf8(buf.to_vec()).unwrap()
}

/// Parse a base62 seed of 1 to 6 characters. `None` if bad or above `u32::MAX`.
pub fn seed_from_base62(s: &str) -> Option<u32> {
    if s.is_empty() || s.len() > 6 {
        return None;
    }
    let mut v: u64 = 0;
    for c in s.bytes() {
        let d = SEED_ALPHABET.iter().position(|&x| x == c)? as u64;
        v = v.checked_mul(62)?.checked_add(d)?;
    }
    if v > u32::MAX as u64 {
        return None;
    }
    Some(v as u32)
}

pub struct XorShift {
    pub(super) s: u32,
}

impl XorShift {
    pub fn new(seed: u32) -> Self {
        Self { s: seed.max(1) }
    }
    pub fn next_u32(&mut self) -> u32 {
        self.s = self.s.wrapping_add(0x6d2b79f5);
        let mut t = self.s;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        t ^ (t >> 14)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) / 4_294_967_296.0
    }
}
