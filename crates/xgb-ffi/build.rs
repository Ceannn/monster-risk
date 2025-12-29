use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=XGBOOST_LIB_DIR");

    let lib_dir = env::var("XGBOOST_LIB_DIR")
        .expect("Set XGBOOST_LIB_DIR to .../site-packages/xgboost/lib (contains libxgboost.so)");
    let lib_dir = PathBuf::from(lib_dir);

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=xgboost");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // lib_dir = .../site-packages/xgboost/lib
    // site_packages = lib_dir.parent().parent()
    if let Some(xgboost_dir) = lib_dir.parent() {
        if let Some(site_packages) = xgboost_dir.parent() {
            let libs_dir = site_packages.join("xgboost.libs");
            if libs_dir.exists() {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libs_dir.display());
            }
        }
    }
}
