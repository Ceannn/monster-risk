use xgb_ffi::Booster;

#[test]
fn smoke_predict() {
    let model = std::env::var("XGB_MODEL").expect("set XGB_MODEL=/path/to/ieee_xgb.ubj");
    let bst = Booster::load_model(model).unwrap();

    let row = vec![f32::NAN; 92];
    let p = bst.predict_proba_dense_1row(&row).unwrap();
    eprintln!("p={p}");
}
