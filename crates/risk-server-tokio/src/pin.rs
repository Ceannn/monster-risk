// crates/risk-server-tokio/src/pin.rs
#![allow(dead_code)]

use anyhow::Context;

#[cfg(target_os = "linux")]
pub fn pin_tokio_runtime_workers(io_cpus: &[usize]) -> anyhow::Result<()> {
    use libc::{cpu_set_t, pid_t, sched_setaffinity, CPU_SET, CPU_ZERO};
    use std::fs;

    anyhow::ensure!(!io_cpus.is_empty(), "io_cpus is empty");

    // build cpu mask
    let mut set: cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe { CPU_ZERO(&mut set) };
    for &cpu in io_cpus {
        unsafe { CPU_SET(cpu, &mut set) };
    }

    let mut pinned = 0usize;
    let mut seen = 0usize;

    for ent in fs::read_dir("/proc/self/task").context("read /proc/self/task")? {
        let ent = ent?;
        let tid_str = ent.file_name().to_string_lossy().to_string();
        let tid: pid_t = tid_str
            .parse::<i32>()
            .with_context(|| format!("parse tid '{tid_str}'"))?;

        let comm_path = ent.path().join("comm");
        let name = std::fs::read_to_string(&comm_path).unwrap_or_default();
        let name = name.trim();

        // 只 pin tokio runtime worker，避免影响 xgb-worker/main/其他线程的 affinity 继承
        if !name.starts_with("tokio-runtime-w") && !name.contains("tokio-runtime") {
            continue;
        }

        seen += 1;
        let rc = unsafe { sched_setaffinity(tid, std::mem::size_of::<cpu_set_t>(), &set) };
        if rc != 0 {
            let e = std::io::Error::last_os_error();
            tracing::warn!(tid, thread=%name, err=%e, "pin tokio runtime worker failed");
            continue;
        }
        pinned += 1;
    }

    tracing::info!(
        io_cpus = ?io_cpus,
        tokio_threads_seen = seen,
        tokio_threads_pinned = pinned,
        "pin_tokio_runtime_workers done"
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn pin_tokio_runtime_workers(_io_cpus: &[usize]) -> anyhow::Result<()> {
    Ok(())
}
