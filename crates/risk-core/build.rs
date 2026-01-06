// TL2CGEN build script: compile tl2cgen-generated C and safely namespace symbols.
//
// Context:
// - We statically link multiple tl2cgen models (L1 + L2) into one Rust binary.
// - tl2cgen emits generic global symbols like `predict`, `get_num_feature`, `is_categorical`.
// - Without namespacing, the linker will see duplicate symbols.
//
// DO NOT do this:
//   objcopy --prefix-symbols=ieee_l2_ libieee_l2.a
// It prefixes *everything*, including *undefined* references (e.g. expf -> ieee_l2_expf),
// which then fails to link unless you add custom shims.
//
// What we do instead:
// - Use `nm -g --defined-only` to list only **defined** global symbols in the archive.
// - Apply `objcopy --redefine-sym old=prefix_old` for those defined symbols only.
// - Undefined references remain untouched and keep linking to system libs (libm, libc...).

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    // Re-run when flags change
    println!("cargo:rerun-if-env-changed=TL2CGEN_MARCH_NATIVE");

    let l1_enabled = env::var_os("CARGO_FEATURE_NATIVE_L1_TL2CGEN").is_some();
    let l2_enabled = env::var_os("CARGO_FEATURE_NATIVE_L2_TL2CGEN").is_some();

    // tl2cgen uses expf/logf in postprocess; explicitly link libm (redundant on many targets,
    // but harmless and prevents surprises with -nodefaultlibs toolchains).
    if l1_enabled || l2_enabled {
        println!("cargo:rustc-link-lib=m");
    }

    if l1_enabled {
        compile_tl2cgen_model("native/tl2cgen/ieee_l1", "ieee_l1", None);
    }

    if l2_enabled {
        // Namespace L2 because we often ship L1+L2 together.
        compile_tl2cgen_model("native/tl2cgen/ieee_l2", "ieee_l2", Some("ieee_l2_"));
    }
}

fn march_native_enabled() -> bool {
    matches!(
        env::var("TL2CGEN_MARCH_NATIVE").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

fn tool_exists(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn find_objcopy() -> &'static str {
    if tool_exists("llvm-objcopy") {
        "llvm-objcopy"
    } else {
        "objcopy"
    }
}

fn find_nm() -> &'static str {
    // GNU nm is typically `nm`; llvm version is `llvm-nm`.
    if tool_exists("llvm-nm") {
        "llvm-nm"
    } else {
        "nm"
    }
}

fn run_checked(mut cmd: Command) {
    let pretty = format!("{:?}", cmd);
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {pretty}: {e}"));
    if !out.status.success() {
        panic!(
            "command failed: {pretty}\nstatus={:?}\nstdout=\n{}\nstderr=\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

fn list_c_files(model_dir: &Path) -> Vec<PathBuf> {
    let mut c_files: Vec<PathBuf> = fs::read_dir(model_dir)
        .unwrap_or_else(|e| panic!("failed to read_dir {model_dir:?}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("c"))
        .collect();

    // Keep build deterministic
    c_files.sort();

    // If you ever drop helper C files next to the generated model, keep them out by default.
    c_files.retain(|p| {
        let name = p.file_name().and_then(|x| x.to_str()).unwrap_or("");
        !name.starts_with("compat_")
    });

    c_files
}

fn nm_defined_globals(archive: &Path) -> Vec<String> {
    let nm = find_nm();
    let out = Command::new(nm)
        .args(["-g", "--defined-only"])
        .arg(archive)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {nm}: {e}"));

    if !out.status.success() {
        panic!(
            "nm failed: {:?}\nstdout=\n{}\nstderr=\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let mut syms = Vec::new();

    for line in text.lines() {
        // archive headers look like: "libfoo.a:obj.o:"; skip
        if line.ends_with(':') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let sym = parts[parts.len() - 1];
        // Skip toolchain/internal symbols just in case.
        if sym.starts_with("__") || sym.starts_with("_GLOBAL_") {
            continue;
        }
        syms.push(sym.to_string());
    }

    syms.sort();
    syms.dedup();
    syms
}

fn namespace_defined_symbols_in_archive(archive: &Path, prefix: &str, must_have: &str) {
    let syms = nm_defined_globals(archive);

    let mut args: Vec<String> = Vec::new();
    for s in syms {
        if s.starts_with(prefix) {
            continue;
        }
        args.push("--redefine-sym".to_string());
        args.push(format!("{s}={prefix}{s}"));
    }

    if args.is_empty() {
        return;
    }

    let objcopy = find_objcopy();
    let mut cmd = Command::new(objcopy);
    for a in args {
        cmd.arg(a);
    }
    cmd.arg(archive);
    run_checked(cmd);

    // Sanity check: ensure we actually produced the expected namespaced symbol.
    let after = nm_defined_globals(archive);
    let want = format!("{prefix}{must_have}");
    if !after.iter().any(|s| s == &want) {
        panic!(
            "namespacing failed: expected symbol {want} not found in archive {archive:?}\n\
             hint: run `nm -g --defined-only {archive:?} | rg {want}` to debug"
        );
    }
}

fn compile_tl2cgen_model(model_dir_rel: &str, lib_name: &str, namespace_prefix: Option<&str>) {
    let model_dir = Path::new(model_dir_rel);
    if !model_dir.exists() {
        panic!("tl2cgen model dir not found: {model_dir:?}");
    }

    // Ensure `cargo` rebuilds if the generated C changes.
    println!("cargo:rerun-if-changed={model_dir_rel}");

    let mut build = cc::Build::new();
    build
        .warnings(false)
        .extra_warnings(false)
        .opt_level(3)
        .debug(false)
        .flag("-ffunction-sections")
        .flag("-fdata-sections")
        .flag("-fno-plt")
        .flag("-fno-exceptions")
        .flag("-fno-unwind-tables")
        .flag("-fno-asynchronous-unwind-tables")
        .flag("-fno-stack-protector")
        .flag("-fomit-frame-pointer");

    if march_native_enabled() {
        build.flag("-march=native");
    }

    // tl2cgen emits C99
    build.flag("-std=c99");

    for f in list_c_files(model_dir) {
        build.file(f);
    }

    // Build as a static archive: lib{lib_name}.a in OUT_DIR
    build.compile(lib_name);

    if let Some(prefix) = namespace_prefix {
        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
        let archive = out_dir.join(format!("lib{lib_name}.a"));
        if !archive.exists() {
            panic!("expected archive not found: {archive:?}");
        }

        // Namespace only *defined* symbols to avoid renaming undefined refs.
        namespace_defined_symbols_in_archive(&archive, prefix, "predict");
    }
}
