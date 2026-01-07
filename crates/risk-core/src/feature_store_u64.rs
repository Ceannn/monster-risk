use dashmap::DashMap;
use std::collections::VecDeque;

/// 100ms 分桶的滑窗特征存储（u64 key 版本）
///
/// 目标：
/// - query_then_update：严格点时一致（当前交易的特征不包含自己）
/// - event_time 语义：以事件时间推进滑窗，而不是 arrival time
/// - 有限乱序容忍：允许小幅 out-of-order；严重乱序会标记并降级
///
/// 设计取舍（方便你写论文/答辩）：
/// - 100ms 桶粒度更细，能捕捉“突发脉冲”
/// - 但窗口内桶数更多（300s → 3000 桶）
/// - 为避免每个用户预分配 3000 桶导致内存炸裂，这里使用 **稀疏 VecDeque**：
///   只为出现过交易的 bucket 分配节点。低频用户非常省内存。
#[derive(Debug, Clone)]
pub struct FeatureStoreU64 {
    users: DashMap<u64, UserState>,
    win_60s: i64,
    win_300s: i64,
}

#[derive(Debug, Clone, Default)]
struct UserState {
    last_event_time_ms: i64,
    max_bucket_seen: i64,
    buckets: VecDeque<Bucket>, // 按 bucket_id 升序
}

#[derive(Debug, Clone, Copy, Default)]
struct Bucket {
    bucket_id: i64, // event_time_ms / BUCKET_MS
    cnt: u32,
    sum: f64,
}

/// 100ms 桶
const BUCKET_MS: i64 = 100;

/// 允许的乱序容忍（ms）：小乱序尽量支持；大乱序标记并降级。
const OOO_TOL_MS: i64 = 2_000;

/// 防御：event_time 被恶意拉到未来导致状态膨胀（可按需调整）
const MAX_FUTURE_SKEW_MS: i64 = 60_000;

#[derive(Debug, Clone, Default)]
pub struct StoreFeatures {
    pub velocity_60s: f64,
    pub velocity_300s: f64,
    pub amount_sum_60s: f64,
    pub amount_sum_300s: f64,
    pub inter_arrival_ms: f64,
    /// 乱序标记：0/1（用 f64 方便直接入特征向量）
    pub oo_order_flag: f64,
}

impl FeatureStoreU64 {
    pub fn new(win_60s: u64, win_300s: u64) -> Self {
        Self {
            users: DashMap::new(),
            win_60s: (win_60s as i64) * 1000,
            win_300s: (win_300s as i64) * 1000,
        }
    }

    #[inline]
    fn win_buckets_60(&self) -> i64 {
        (self.win_60s / BUCKET_MS).max(1)
    }

    #[inline]
    fn win_buckets_300(&self) -> i64 {
        (self.win_300s / BUCKET_MS).max(1)
    }

    /// 核心：先 query 再 update，保证“这笔交易的特征不包含它自己”。
    pub fn query_then_update(&self, key: u64, event_time_ms: i64, amount: f64) -> StoreFeatures {
        let mut entry = self.users.entry(key).or_default();

        // --- 基本防御：避免 event_time 过于离谱（未来太多会导致窗口推进大量空桶）
        let now_ms = current_time_ms();
        let mut t_ms = event_time_ms;
        if t_ms > now_ms + MAX_FUTURE_SKEW_MS {
            t_ms = now_ms + MAX_FUTURE_SKEW_MS;
        }

        let event_bucket = t_ms.div_euclid(BUCKET_MS);

        // --- 乱序判定
        let mut oo_flag = 0.0;
        let mut severe_oo = false;
        let inter_arrival = if entry.last_event_time_ms > 0 {
            let dt = t_ms - entry.last_event_time_ms;
            if dt < 0 {
                oo_flag = 1.0;
                if -dt > OOO_TOL_MS {
                    severe_oo = true;
                }
                0.0
            } else {
                dt as f64
            }
        } else {
            0.0
        };

        // --- 维护“全局推进”的 max_bucket_seen，用来做内存过期清理
        if entry.max_bucket_seen == 0 {
            entry.max_bucket_seen = event_bucket;
        } else if event_bucket > entry.max_bucket_seen {
            entry.max_bucket_seen = event_bucket;
        }

        // --- 过期清理：基于 max_bucket_seen（而不是 event_bucket）
        self.expire_old(&mut entry);

        // --- query：以 event_bucket 为“点时”，只统计 <= event_bucket 的桶
        let (cnt_60, sum_60, cnt_300, sum_300) = self.query_windows(&entry, event_bucket);

        let features = StoreFeatures {
            velocity_60s: cnt_60 as f64,
            velocity_300s: cnt_300 as f64,
            amount_sum_60s: sum_60,
            amount_sum_300s: sum_300,
            inter_arrival_ms: inter_arrival,
            oo_order_flag: oo_flag,
        };

        // --- update：严重乱序就不写入（否则相当于“回填历史”，会让状态不一致）
        if !severe_oo {
            self.upsert_bucket(&mut entry, event_bucket, amount);
        }

        // last_event_time_ms 只做单调推进（保证正常流量下 inter-arrival 合理）
        if t_ms > entry.last_event_time_ms {
            entry.last_event_time_ms = t_ms;
        }

        features
    }

    fn query_windows(&self, st: &UserState, event_bucket: i64) -> (u32, f64, u32, f64) {
        let w60 = self.win_buckets_60();
        let w300 = self.win_buckets_300();

        let min_60 = event_bucket - w60 + 1;
        let min_300 = event_bucket - w300 + 1;

        let mut c60: u32 = 0;
        let mut s60: f64 = 0.0;
        let mut c300: u32 = 0;
        let mut s300: f64 = 0.0;

        for b in st.buckets.iter() {
            if b.bucket_id > event_bucket {
                // “未来桶”（相对这笔 event_time）不计入点时统计
                break;
            }
            if b.bucket_id >= min_300 {
                c300 = c300.saturating_add(b.cnt);
                s300 += b.sum;
            }
            if b.bucket_id >= min_60 {
                c60 = c60.saturating_add(b.cnt);
                s60 += b.sum;
            }
        }

        (c60, s60, c300, s300)
    }

    fn expire_old(&self, st: &mut UserState) {
        let w300 = self.win_buckets_300();
        let min_keep = st.max_bucket_seen - w300 + 1;

        while let Some(front) = st.buckets.front() {
            if front.bucket_id < min_keep {
                st.buckets.pop_front();
            } else {
                break;
            }
        }
    }

    fn upsert_bucket(&self, st: &mut UserState, bucket_id: i64, amount: f64) {
        // 空：直接 push
        if st.buckets.is_empty() {
            st.buckets.push_back(Bucket {
                bucket_id,
                cnt: 1,
                sum: amount,
            });
            return;
        }

        // 快路径：落在末尾（最常见）
        if let Some(last) = st.buckets.back_mut() {
            if last.bucket_id == bucket_id {
                last.cnt = last.cnt.saturating_add(1);
                last.sum += amount;
                return;
            } else if last.bucket_id < bucket_id {
                st.buckets.push_back(Bucket {
                    bucket_id,
                    cnt: 1,
                    sum: amount,
                });
                return;
            }
        }

        // 慢路径：小乱序，插入/更新中间桶
        let search = {
            let slice = st.buckets.make_contiguous();
            slice.binary_search_by_key(&bucket_id, |b| b.bucket_id)
        };

        match search {
            Ok(i) => {
                let b = &mut st.buckets[i];
                b.cnt = b.cnt.saturating_add(1);
                b.sum += amount;
            }
            Err(i) => {
                st.buckets.insert(
                    i,
                    Bucket {
                        bucket_id,
                        cnt: 1,
                        sum: amount,
                    },
                );
            }
        }
    }
}

fn current_time_ms() -> i64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    now.as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_then_update_point_in_time() {
        let store = FeatureStoreU64::new(60, 300);
        let uid: u64 = 1;

        // 第 1 笔：query 应是空
        let f1 = store.query_then_update(uid, 1_000, 10.0);
        assert_eq!(f1.velocity_60s as u32, 0);
        assert!((f1.amount_sum_300s - 0.0).abs() < 1e-9);

        // 第 2 笔：query 应能看到第 1 笔（但不包含自己）
        let f2 = store.query_then_update(uid, 1_050, 5.0);
        assert_eq!(f2.velocity_60s as u32, 1);
        assert!((f2.amount_sum_300s - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_window_expire_60_keep_300() {
        let store = FeatureStoreU64::new(60, 300);
        let uid: u64 = 2;

        // 在 t=0 插入一笔
        let _ = store.query_then_update(uid, 0, 7.0);

        // 60.1s 后再来一笔：query 时 60s 窗口应过期，300s 窗口仍保留
        let f = store.query_then_update(uid, 60_100, 1.0);
        assert_eq!(f.velocity_60s as u32, 0);
        assert_eq!(f.velocity_300s as u32, 1);
        assert!((f.amount_sum_300s - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_severe_out_of_order_degrades_update() {
        let store = FeatureStoreU64::new(60, 300);
        let uid: u64 = 3;

        let _ = store.query_then_update(uid, 10_000, 10.0);
        // 严重乱序：比 last_event_time 早 3s（> 2s 容忍）
        let f_oo = store.query_then_update(uid, 7_000, 5.0);
        assert_eq!(f_oo.oo_order_flag as u32, 1);

        // 再来一个正常事件：不应看到那笔严重乱序被写入
        let f_next = store.query_then_update(uid, 10_100, 1.0);
        assert!((f_next.amount_sum_300s - 10.0).abs() < 1e-9);
    }
}
