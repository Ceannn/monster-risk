use std::cell::{Cell, RefCell};
use std::sync::OnceLock;
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

static METRICS_SAMPLE_LOG2: OnceLock<u32> = OnceLock::new();
static METRICS_COUNTER_BATCH: OnceLock<u64> = OnceLock::new();
static METRICS_COUNTER_FLUSH_MS: OnceLock<u64> = OnceLock::new();
static METRICS_COUNTER_START: OnceLock<Instant> = OnceLock::new();

thread_local! {
    static METRICS_SAMPLE_COUNTER: Cell<u64> = const { Cell::new(0) };
}

thread_local! {
    static METRICS_COUNTER_BUF: RefCell<CounterBuf> = RefCell::new(CounterBuf::new());
}

#[inline]
pub fn metrics_sample_log2() -> u32 {
    *METRICS_SAMPLE_LOG2.get_or_init(|| {
        std::env::var("RISK_METRICS_SAMPLE_LOG2")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0)
            .min(30)
    })
}

#[inline]
pub fn metrics_should_sample() -> bool {
    let log2 = metrics_sample_log2();
    if log2 == 0 {
        return true;
    }
    if log2 >= 63 {
        return false;
    }
    let mask = (1u64 << log2) - 1;
    METRICS_SAMPLE_COUNTER.with(|c| {
        let n = c.get().wrapping_add(1);
        c.set(n);
        (n & mask) == 0
    })
}

#[inline]
fn metrics_counter_batch() -> u64 {
    *METRICS_COUNTER_BATCH.get_or_init(|| {
        std::env::var("RISK_METRICS_COUNTER_BATCH")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(256)
            .max(1)
    })
}

#[inline]
fn metrics_counter_flush_ms() -> u64 {
    *METRICS_COUNTER_FLUSH_MS.get_or_init(|| {
        std::env::var("RISK_METRICS_COUNTER_FLUSH_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(200)
    })
}

#[inline]
fn metrics_counter_now_ms() -> u64 {
    METRICS_COUNTER_START
        .get_or_init(Instant::now)
        .elapsed()
        .as_millis() as u64
}

#[derive(Debug, Clone, Copy)]
struct CounterSlot {
    name: &'static str,
    count: u64,
}

#[derive(Debug)]
struct CounterBuf {
    slots: Vec<CounterSlot>,
    total: u64,
    last_flush_ms: u64,
}

impl CounterBuf {
    fn new() -> Self {
        Self {
            slots: Vec::with_capacity(16),
            total: 0,
            last_flush_ms: 0,
        }
    }

    fn flush(&mut self, now_ms: u64) {
        for slot in &mut self.slots {
            if slot.count > 0 {
                metrics::counter!(slot.name).increment(slot.count);
                slot.count = 0;
            }
        }
        self.total = 0;
        self.last_flush_ms = now_ms;
    }
}

pub fn metrics_counter_add(name: &'static str, delta: u64) {
    if delta == 0 {
        metrics::counter!(name).increment(0);
        return;
    }

    let batch = metrics_counter_batch();
    let flush_ms = metrics_counter_flush_ms();
    if batch <= 1 && flush_ms == 0 {
        metrics::counter!(name).increment(delta);
        return;
    }

    METRICS_COUNTER_BUF.with(|buf| {
        let mut buf = buf.borrow_mut();
        let mut found = false;
        for slot in &mut buf.slots {
            if slot.name == name {
                slot.count = slot.count.saturating_add(delta);
                found = true;
                break;
            }
        }
        if !found {
            buf.slots.push(CounterSlot { name, count: delta });
        }
        buf.total = buf.total.saturating_add(1);

        if buf.total >= batch {
            let now_ms = metrics_counter_now_ms();
            buf.flush(now_ms);
            return;
        }

        if flush_ms > 0 && (buf.total & 0x3f) == 0 {
            let now_ms = metrics_counter_now_ms();
            if now_ms.saturating_sub(buf.last_flush_ms) >= flush_ms {
                buf.flush(now_ms);
            }
        }
    });
}

#[derive(Clone, Copy)]
pub struct BatchedCounter {
    name: &'static str,
}

impl BatchedCounter {
    #[inline]
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }

    #[inline]
    pub fn increment(&self, value: u64) {
        metrics_counter_add(self.name, value);
    }
}

#[derive(Clone, Copy)]
pub struct SampledHistogram {
    name: &'static str,
    enabled: bool,
}

impl SampledHistogram {
    #[inline]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            enabled: metrics_should_sample(),
        }
    }

    #[inline]
    pub fn record(&self, value: f64) {
        if self.enabled {
            metrics::histogram!(self.name).record(value);
        }
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled
    }
}

#[macro_export]
macro_rules! sampled_histogram {
    ($name:literal) => {{
        $crate::util::SampledHistogram::new($name)
    }};
    ($name:literal, $value:expr) => {{
        if $crate::util::metrics_should_sample() {
            metrics::histogram!($name).record($value as f64);
        }
    }};
}

#[macro_export]
macro_rules! batched_counter {
    ($name:literal) => {{
        $crate::util::BatchedCounter::new($name)
    }};
}
