use crate::config::Config;
use crate::feature_store_u64::{FeatureStoreU64, StoreFeatures};
use anyhow::{bail, Context};
use bytes::{Bytes, BytesMut};
use crate::util::mix_u64;
use hashbrown::HashMap;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

pub const EXTRA_FEATURE_NAMES: &[&str] = &[
    "uid_velocity_60s",
    "uid_velocity_300s",
    "uid_sum_amt_60s",
    "uid_sum_amt_300s",
    "uid_avg_amt_60s",
    "uid_avg_amt_300s",
    "uid_inter_arrival_ms",
    "uid_ooo_flag",
    "ip_velocity_60s",
    "ip_velocity_300s",
    "ip_sum_amt_60s",
    "ip_sum_amt_300s",
    "ip_avg_amt_60s",
    "ip_avg_amt_300s",
    "ip_inter_arrival_ms",
    "ip_ooo_flag",
    "dev_velocity_60s",
    "dev_velocity_300s",
    "dev_sum_amt_60s",
    "dev_sum_amt_300s",
    "dev_avg_amt_60s",
    "dev_avg_amt_300s",
    "dev_inter_arrival_ms",
    "dev_ooo_flag",
    "uniq_uid_per_ip_60s",
    "uniq_uid_per_ip_300s",
    "uniq_ip_per_uid_60s",
    "uniq_ip_per_uid_300s",
];

pub const EXTRA_FEATURE_DIM: usize = 28;

#[derive(Debug, Clone, Copy, Default)]
pub struct DenseStatefulTiming {
    pub pre_l1_us: u64,
    pub build_l2_payload_us: u64,
}

#[derive(Debug, Clone)]
pub struct DenseStatefulCtx {
    pub extra: [f32; EXTRA_FEATURE_DIM],
}

#[derive(Debug, Clone)]
pub struct StatefulL2Augmenter {
    l1_dim: usize,
    l2_dim: usize,
    idx_dt: usize,
    idx_amt: usize,
    idx_uid: usize,
    idx_ip: usize,
    idx_dev: usize,
    store_uid: FeatureStoreU64,
    store_ip: FeatureStoreU64,
    store_dev: FeatureStoreU64,
    graph_uid_per_ip: BipartiteUniqueStore,
    graph_ip_per_uid: BipartiteUniqueStore,
}

impl StatefulL2Augmenter {
    pub fn try_new(cfg: &Config, l1_feature_names: &[String], l2_feature_names: &[String]) -> anyhow::Result<Option<Self>> {
        let l1_dim = l1_feature_names.len();
        let l2_dim = l2_feature_names.len();
        if l2_dim == l1_dim {
            return Ok(None);
        }
        if l2_dim != l1_dim + EXTRA_FEATURE_DIM {
            bail!(
                "L2 feature dim mismatch: l1_dim={} l2_dim={} expect={} (L1 + {})",
                l1_dim,
                l2_dim,
                l1_dim + EXTRA_FEATURE_DIM,
                EXTRA_FEATURE_DIM
            );
        }
        for i in 0..l1_dim {
            if l1_feature_names[i] != l2_feature_names[i] {
                bail!(
                    "L2 feature_names prefix mismatch at idx={}: l1='{}' l2='{}'",
                    i,
                    l1_feature_names[i],
                    l2_feature_names[i]
                );
            }
        }
        for (j, name) in EXTRA_FEATURE_NAMES.iter().enumerate() {
            let got = &l2_feature_names[l1_dim + j];
            if got != name {
                bail!(
                    "L2 extra feature_names mismatch at idx={} expect='{}' got='{}'",
                    l1_dim + j,
                    name,
                    got
                );
            }
        }

        let idx_dt = find_idx(l1_feature_names, "TransactionDT")
            .context("stateful L2 requires feature TransactionDT")?;
        let idx_amt = find_idx(l1_feature_names, "TransactionAmt")
            .context("stateful L2 requires feature TransactionAmt")?;
        let idx_uid = find_idx(l1_feature_names, "card1")
            .context("stateful L2 requires feature card1")?;
        let idx_ip = find_idx(l1_feature_names, "addr1")
            .context("stateful L2 requires feature addr1")?;
        let idx_dev = find_idx(l1_feature_names, "DeviceInfo")
            .or_else(|| find_idx(l1_feature_names, "DeviceType"))
            .context("stateful L2 requires feature DeviceInfo or DeviceType")?;

        Ok(Some(Self {
            l1_dim,
            l2_dim,
            idx_dt,
            idx_amt,
            idx_uid,
            idx_ip,
            idx_dev,
            store_uid: FeatureStoreU64::new(cfg.win_60s, cfg.win_300s),
            store_ip: FeatureStoreU64::new(cfg.win_60s, cfg.win_300s),
            store_dev: FeatureStoreU64::new(cfg.win_60s, cfg.win_300s),
            graph_uid_per_ip: BipartiteUniqueStore::new(cfg.win_60s, cfg.win_300s),
            graph_ip_per_uid: BipartiteUniqueStore::new(cfg.win_60s, cfg.win_300s),
        }))
    }

    pub fn l2_dim(&self) -> usize {
        self.l2_dim
    }

    pub fn extra_dim(&self) -> usize {
        EXTRA_FEATURE_DIM
    }

    pub fn pre_l1(&self, payload: &Bytes) -> (DenseStatefulCtx, DenseStatefulTiming) {
        let t0 = Instant::now();
        let dt_s = read_f32_le(payload, self.idx_dt).unwrap_or(0.0);
        let amt_f32 = read_f32_le(payload, self.idx_amt).unwrap_or(0.0);
        let uid_f32 = read_f32_le(payload, self.idx_uid).unwrap_or(f32::NAN);
        let ip_f32 = read_f32_le(payload, self.idx_ip).unwrap_or(f32::NAN);
        let dev_f32 = read_f32_le(payload, self.idx_dev).unwrap_or(f32::NAN);

        let ts_ms = ((dt_s as f64) * 1000.0).round() as i64;
        let amt = amt_f32 as f64;
        let uid = key_from_f32(uid_f32);
        let ip = key_from_f32(ip_f32);
        let dev = key_from_f32(dev_f32);

        let uid_sf = if uid != 0 {
            self.store_uid.query_then_update(uid, ts_ms, amt)
        } else {
            StoreFeatures::default()
        };
        let ip_sf = if ip != 0 {
            self.store_ip.query_then_update(ip, ts_ms, amt)
        } else {
            StoreFeatures::default()
        };
        let dev_sf = if dev != 0 {
            self.store_dev.query_then_update(dev, ts_ms, amt)
        } else {
            StoreFeatures::default()
        };

        let uniq_uid_per_ip = if uid != 0 && ip != 0 {
            self.graph_uid_per_ip.query_then_update(ip, uid, ts_ms)
        } else {
            (0.0f32, 0.0f32)
        };
        let uniq_ip_per_uid = if uid != 0 && ip != 0 {
            self.graph_ip_per_uid.query_then_update(uid, ip, ts_ms)
        } else {
            (0.0f32, 0.0f32)
        };

        let mut extra = [0.0f32; EXTRA_FEATURE_DIM];
        let mut k = 0usize;
        push_key_features(&mut extra, &mut k, &uid_sf);
        push_key_features(&mut extra, &mut k, &ip_sf);
        push_key_features(&mut extra, &mut k, &dev_sf);
        extra[k] = uniq_uid_per_ip.0;
        extra[k + 1] = uniq_uid_per_ip.1;
        extra[k + 2] = uniq_ip_per_uid.0;
        extra[k + 3] = uniq_ip_per_uid.1;

        let pre_l1_us = t0.elapsed().as_micros() as u64;
        (
            DenseStatefulCtx { extra },
            DenseStatefulTiming {
                pre_l1_us,
                build_l2_payload_us: 0,
            },
        )
    }

    /// Deprecated path: allocates a new Bytes buffer. Prefer worker-side extra append.
    pub fn build_l2_payload(&self, payload_l1: &Bytes, ctx: &DenseStatefulCtx) -> (Bytes, DenseStatefulTiming) {
        let t0 = Instant::now();
        debug_assert_eq!(payload_l1.len(), self.l1_dim * 4);
        let mut out = BytesMut::with_capacity(payload_l1.len() + EXTRA_FEATURE_DIM * 4);
        out.extend_from_slice(payload_l1);
        for v in ctx.extra.iter() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let build_l2_payload_us = t0.elapsed().as_micros() as u64;
        (
            out.freeze(),
            DenseStatefulTiming {
                pre_l1_us: 0,
                build_l2_payload_us,
            },
        )
    }
}

fn push_key_features(dst: &mut [f32; EXTRA_FEATURE_DIM], idx: &mut usize, sf: &StoreFeatures) {
    let v60 = sf.velocity_60s as f32;
    let v300 = sf.velocity_300s as f32;
    let s60 = sf.amount_sum_60s as f32;
    let s300 = sf.amount_sum_300s as f32;
    let a60 = if sf.velocity_60s > 0.0 {
        (sf.amount_sum_60s / sf.velocity_60s) as f32
    } else {
        0.0
    };
    let a300 = if sf.velocity_300s > 0.0 {
        (sf.amount_sum_300s / sf.velocity_300s) as f32
    } else {
        0.0
    };
    let ia = sf.inter_arrival_ms as f32;
    let ooo = sf.oo_order_flag as f32;

    dst[*idx] = v60;
    dst[*idx + 1] = v300;
    dst[*idx + 2] = s60;
    dst[*idx + 3] = s300;
    dst[*idx + 4] = a60;
    dst[*idx + 5] = a300;
    dst[*idx + 6] = ia;
    dst[*idx + 7] = ooo;

    *idx += 8;
}

fn find_idx(names: &[String], target: &str) -> Option<usize> {
    names.iter().position(|s| s == target)
}

fn key_from_f32(v: f32) -> u64 {
    if !v.is_finite() {
        return 0;
    }
    // IEEE 类别特征多数是整数编码。对负数/异常值做保护。
    let i = v.round() as i64;
    if i <= 0 {
        0
    } else {
        i as u64
    }
}

fn read_f32_le(bytes: &Bytes, idx: usize) -> Option<f32> {
    let off = idx.checked_mul(4)?;
    let end = off.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    let b = &bytes[off..end];
    let raw = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    Some(f32::from_bits(raw))
}

#[derive(Debug, Default)]
struct UniqueWindow {
    win_ms: i64,
    max_events: usize,
    q: VecDeque<(i64, u64)>,
    counts: HashMap<u64, u32>,
}

impl UniqueWindow {
    fn new(win_ms: i64, max_events: usize) -> Self {
        Self {
            win_ms,
            max_events,
            q: VecDeque::new(),
            counts: HashMap::new(),
        }
    }

    fn expire(&mut self, now_ms: i64) {
        let min_keep = now_ms.saturating_sub(self.win_ms);
        while let Some((t, other)) = self.q.front().copied() {
            if t >= min_keep {
                break;
            }
            self.q.pop_front();
            if let Some(c) = self.counts.get_mut(&other) {
                if *c <= 1 {
                    self.counts.remove(&other);
                } else {
                    *c -= 1;
                }
            }
        }
    }

    fn uniq(&self) -> usize {
        self.counts.len()
    }

    fn push(&mut self, now_ms: i64, other: u64) {
        if self.q.len() >= self.max_events {
            // 硬上限：丢弃最早的事件，保证内存不爆。
            // 这会轻微影响计数精度，但对 demo/论文足够。
            if let Some((_, old_other)) = self.q.pop_front() {
                if let Some(c) = self.counts.get_mut(&old_other) {
                    if *c <= 1 {
                        self.counts.remove(&old_other);
                    } else {
                        *c -= 1;
                    }
                }
            }
        }
        self.q.push_back((now_ms, other));
        *self.counts.entry(other).or_insert(0) += 1;
    }
}

#[derive(Debug, Default)]
struct BipartiteKeyState {
    w60: UniqueWindow,
    w300: UniqueWindow,
}

#[derive(Debug, Clone)]
struct BipartiteUniqueStore {
    win_60s_ms: i64,
    win_300s_ms: i64,
    shards: Arc<Vec<Mutex<HashMap<u64, BipartiteKeyState>>>>,
    shard_mask: usize,
}

impl BipartiteUniqueStore {
    fn new(win_60s: u64, win_300s: u64) -> Self {
        let avail = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let default_shards = avail.saturating_mul(2).max(1);
        let shard_hint = std::env::var("FEATURE_STORE_SHARDS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(default_shards);
        let shard_count = shard_hint.max(1).next_power_of_two();
        let mut shards = Vec::with_capacity(shard_count);
        for _ in 0..shard_count {
            shards.push(Mutex::new(HashMap::new()));
        }
        Self {
            win_60s_ms: (win_60s as i64) * 1000,
            win_300s_ms: (win_300s as i64) * 1000,
            shards: Arc::new(shards),
            shard_mask: shard_count - 1,
        }
    }

    fn query_then_update(&self, key: u64, other: u64, now_ms: i64) -> (f32, f32) {
        const MAX_EVENTS_PER_WIN: usize = 4096;
        let shard_idx = (mix_u64(key) as usize) & self.shard_mask;
        let mut shard = self.shards[shard_idx].lock();
        let st = shard.entry(key).or_insert_with(|| BipartiteKeyState {
            w60: UniqueWindow::new(self.win_60s_ms, MAX_EVENTS_PER_WIN),
            w300: UniqueWindow::new(self.win_300s_ms, MAX_EVENTS_PER_WIN),
        });

        st.w60.expire(now_ms);
        st.w300.expire(now_ms);
        let u60 = st.w60.uniq() as f32;
        let u300 = st.w300.uniq() as f32;
        st.w60.push(now_ms, other);
        st.w300.push(now_ms, other);
        (u60, u300)
    }
}
