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
