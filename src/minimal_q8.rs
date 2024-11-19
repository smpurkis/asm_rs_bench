#![feature(portable_simd)] // Enable std::arch module for intrinsics

extern crate blas_sys;
extern crate blis_src;

mod quantization;

use quantization::*;
use rand::prelude::*;
use std::time::Instant;

fn gemm_q8_0_q8_0_scalar(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[BlockQ8_0],
    vy: &[BlockQ8_0],
    nr: usize,
    nc: usize,
) {
    let qk = QK8_0;
    let nb = n / qk;

    assert!(n % qk == 0);

    // Process each row
    for ir in 0..nr {
        // Process each column
        for ic in 0..nc {
            let mut sum = 0.0f32;

            // Process blocks of quantized values
            for ib in 0..nb {
                // sum += vec_dot_q8(&vx[ir * nb + ib], &vy[ic * nb + ib]);
                let vx_block = &vx[ir * nb + ib];
                let vy_block = &vy[ic * nb + ib];

                let scale_x = vx_block.d.to_f32();
                let scale_y = vy_block.d.to_f32();

                for i in 0..QK8_0 {
                    let x_val = vx_block.qs[i];
                    let y_val = vy_block.qs[i];
                    sum += scale_x * scale_y * (x_val as f32) * (y_val as f32);
                }
            }

            // Store result
            s[ir * bs + ic] = sum;
        }
    }
}

// Unquantized GEMV function
fn gemm_unquantized(n: usize, s: &mut [f32], vx: &[f32], vy: &[f32], nr: usize, nc: usize) {
    for i in 0..nr {
        for j in 0..nc {
            for k in 0..n {
                s[i * nc + j] += vx[i * n + k] * vy[j * n + k];
            }
        }
    }
}

fn benchmark_q8_implementation(
    iterations: usize,
    n: usize,
    nc: usize,
    nr: usize,
    bs: usize,
    test_data: &TestData,
) {
    let name = "Quantized GEMM q8";

    // Prepare variables
    let mut s_quantized = vec![0.0f32; nc * nr];
    let mut s_unquantized = vec![0.0f32; nc * nr];

    // Benchmark quantized GEMV (scalar)
    let start_q = Instant::now();
    for _ in 0..iterations {
        gemm_q8_0_q8_0_scalar(
            n,
            &mut s_quantized,
            bs,
            &test_data.vx_80,
            &test_data.vy_80,
            nr,
            nc,
        );
    }
    let duration_q = start_q.elapsed();

    // Benchmark unquantized GEMV
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemm_unquantized(n, &mut s_unquantized, &test_data.ux, &test_data.uy, nr, nc);
    }
    let duration_uq = start_uq.elapsed();

    println!();

    println!(
        "{} (Quantized): {:.2} ms per iteration",
        name,
        (duration_q.as_secs_f64() * 1000.0) / iterations as f64
    );

    println!(
        "Unquantized GEMV (Unquantized): {:.2} ms per iteration",
        (duration_uq.as_secs_f64() * 1000.0) / iterations as f64
    );

    // Compare outputs
    println!("Comparing outputs between quantized and unquantized GEMM:");
    let max_diff = s_quantized
        .iter()
        .zip(s_unquantized.iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .fold(0.0, f64::max);
    println!("Maximum difference: {:.6}", max_diff);

    println!(
        "s_unquantized q8: {:?}",
        s_unquantized.to_vec()[s_unquantized.len() - 10..s_unquantized.len()].to_vec()
    );
    println!(
        "s_quantized q8: {:?}",
        s_quantized.to_vec()[s_quantized.len() - 10..s_quantized.len()].to_vec()
    );
    println!(
        "diff: {:?}",
        s_unquantized
            .iter()
            .zip(s_quantized.iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<f32>>()[(s_unquantized.len() - 10)..s_unquantized.len()]
            .to_vec()
    );
}

// Struct to hold test data
#[derive(Clone, Debug)]
struct TestData {
    vx_80: Vec<BlockQ8_0>,
    vy_80: Vec<BlockQ8_0>,
    ux: Vec<f32>,
    uy: Vec<f32>,
}
// Function to generate test data
fn generate_test_data(n: usize, nc: usize, nr: usize, random: bool) -> TestData {
    // For A matrix (vy)
    let ux_size = nr * n; // A is nr rows x n cols
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let ux: Vec<f32> = (0..ux_size)
        .map(|_| {
            if random {
                rng.gen_range(-0.5..0.5)
            } else {
                1.0f32
            }
        }) // Use actual random data or any desired values
        .collect();

    // For B matrix (vx)
    let uy_size = n * nc; // B is n rows x nc cols

    let uy: Vec<f32> = (0..uy_size)
        .map(|_| {
            if random {
                rand::random::<f32>() - 0.5
            } else {
                1.0f32
            }
        }) // Use actual random data or any desired values
        .collect();

    let vx_80 = quantize_f32_to_q8_0(&ux, n, nr);
    let vy_80 = quantize_f32_to_q8_0(&uy, n, nc);

    TestData {
        vx_80,
        vy_80,
        ux,
        uy,
    }
}

fn main() {
    let n = 32 * 32; // k
    let nc = 32 * 32; // n
    let nr = 4; // m
    let bs = nc;
    let iterations = 1;

    println!(
        "Benchmarking with n={}, nc={}, nr={}, iterations={}",
        n, nc, nr, iterations
    );

    // Generate test data
    let test_data = generate_test_data(n, nc, nr, true);

    // benchmark q8
    benchmark_q8_implementation(iterations, n, nc, nr, bs, &test_data);
}
