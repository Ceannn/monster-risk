.PHONY: run bench fmt clippy

run:
	cargo run -p risk-server-tokio

bench:
	cargo run -p risk-bench -- --rps 1000 --duration 20

fmt:
	cargo fmt

clippy:
	cargo clippy --all-targets --all-features -D warnings
