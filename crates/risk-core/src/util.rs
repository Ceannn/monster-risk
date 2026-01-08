use std::time::{Duration, Instant};

#[inline]
pub fn now_us(start: Instant) -> u64 {
    start.elapsed().as_micros() as u64
}

#[inline]
pub fn dur_us(d: Duration) -> u64 {
    d.as_micros() as u64
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn clamp01(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x }
}

#[inline]
pub fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}
