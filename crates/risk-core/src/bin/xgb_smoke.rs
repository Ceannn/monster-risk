use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // 1) 模型目录：按你实际路径改
    let model_dir = PathBuf::from("models/ieee_cis_xgb");

    // 2) 载入 runtime（会自动选择 ubj > json > bin）
    let rt = risk_core::xgb_runtime::XgbRuntime::load_from_dir(&model_dir)?;

    // 3) 构造一个最小输入：只给几个字段，其余特征会走 missing/NaN
    let obj = serde_json::json!({
        "TransactionAmt": 123.4,
        "ProductCD": "W",
        "card1": 12345
    });
    let obj = obj.as_object().unwrap().clone();

    // 4) build row + predict
    let row = rt.build_row(&obj);
    println!("row_len={}", row.len());

    let p = rt.predict_proba(&row)?;
    println!("pred_proba={}", p);

    // 5) contrib top-k（这里就验证你刚扩展的 xgb_ffi）
    let top = rt.topk_contrib(&row, 8)?;
    println!("topk_contrib:");
    for (name, c) in top {
        println!("  {:<24} {:>10.6}", name, c);
    }

    Ok(())
}
