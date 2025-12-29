// 先最小占位，确保 crate 有 target 能编译起来。
// 你把你写好的 FFI 内容贴到这里，或者用 mod 引入都行。

mod xgb_ffi;
pub use xgb_ffi::*;
