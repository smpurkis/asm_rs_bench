# asm_rs_bench

To run the benchmarks, you need to have `cargo` installed and be on an arm platform. Then, you can run the following command:

```bash
cargo run --release --bin gemv # to run gemv benchmark
cargo run --release --bin gemm # to run gemm benchmark
```

The assembly code is taken from llama.cpp.

Note: I'm currently investigating a bug in the input for the quantized functions that means the outputs are off.
