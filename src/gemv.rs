#![feature(portable_simd)] // Enable std::arch module for intrinsics
extern crate blas_sys;
extern crate blis_src;

mod quantization;

use blas::dgemv;
use blas_sys::sgemv_;
use num::SimdInt;
use quantization::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::simd::*;
use std::time::Instant;

fn gemv_q4_0_4x4_q8_0_scalar(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[BlockQ4_0x4],
    vy: &[BlockQ8_0],
    nr: usize,
    nc: usize,
) {
    let qk = QK8_0;
    let nb = n / qk;

    assert!(n % qk == 0);
    assert!(nc % NCOLS_INTERLEAVED == 0);

    for x in 0..(nc / NCOLS_INTERLEAVED) {
        let mut sumf = [0.0f32; NCOLS_INTERLEAVED];

        for l in 0..nb {
            let b_block = &vx[x * nb + l];
            let a_block = &vy[l];

            for k in 0..(qk / (2 * BLOCK_SIZE)) {
                for j in 0..NCOLS_INTERLEAVED {
                    let mut sumi = 0i32;

                    for i in 0..BLOCK_SIZE {
                        let idx = k * NCOLS_INTERLEAVED * BLOCK_SIZE + j * BLOCK_SIZE + i;

                        let byte = b_block.qs[idx];

                        let v0 = (byte << 4) as i32;
                        let v1 = (byte & 0xF0) as i32;

                        let a_idx0 = k * BLOCK_SIZE + i;
                        let a_idx1 = k * BLOCK_SIZE + i + qk / 2;

                        let a0 = a_block.qs[a_idx0];
                        let a1 = a_block.qs[a_idx1];

                        // Multiply and accumulate with right shift applied to each product
                        sumi += ((v0 * a0 as i32) >> 4) + ((v1 * a1 as i32) >> 4);
                    }

                    let b_scale = b_block.d[j].to_f32();
                    let a_scale = a_block.d.to_f32();

                    sumf[j] += (sumi as f32) * b_scale * a_scale;
                }
            }
        }

        // Store results
        for j in 0..NCOLS_INTERLEAVED {
            s[x * NCOLS_INTERLEAVED + j] = sumf[j];
        }
    }
}

// Parallel GEMV function for quantized data using Rayon
fn gemv_q4_0_4x4_q8_0_scalar_parallel(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[BlockQ4_0x4],
    vy: &[BlockQ8_0],
    nr: usize,
    nc: usize,
) {
    let qk = QK8_0;
    let nb = n / qk;

    assert!(n % qk == 0);
    assert!(nc % NCOLS_INTERLEAVED == 0);

    // Use Rayon's parallel iterator to process chunks
    s.par_chunks_mut(NCOLS_INTERLEAVED)
        .enumerate()
        .for_each(|(x, chunk)| {
            let mut sumf = [0.0f32; NCOLS_INTERLEAVED];

            // Process blocks in parallel within each chunk
            (0..nb).for_each(|l| {
                let b_block = &vx[x * nb + l];
                let a_block = &vy[l];

                for k in 0..(qk / (2 * BLOCK_SIZE)) {
                    for j in 0..NCOLS_INTERLEAVED {
                        let mut sumi = 0i32;

                        for i in 0..BLOCK_SIZE {
                            let idx = k * NCOLS_INTERLEAVED * BLOCK_SIZE + j * BLOCK_SIZE + i;

                            let byte = b_block.qs[idx];
                            let v0 = (byte << 4) as i32;
                            let v1 = (byte & 0xF0) as i32;

                            let a_idx0 = k * BLOCK_SIZE + i;
                            let a_idx1 = k * BLOCK_SIZE + i + qk / 2;

                            let a0 = a_block.qs[a_idx0];
                            let a1 = a_block.qs[a_idx1];

                            // Multiply and accumulate with right shift applied to each product
                            sumi += ((v0 * a0 as i32) >> 4) + ((v1 * a1 as i32) >> 4);
                        }

                        let b_scale = b_block.d[j].to_f32();
                        let a_scale = a_block.d.to_f32();

                        sumf[j] += (sumi as f32) * b_scale * a_scale;
                    }
                }
            });

            // Store results
            chunk.copy_from_slice(&sumf);
        });
}

fn gemv_q4_0_4x4_q8_0_simd(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[BlockQ4_0x4],
    vy: &[BlockQ8_0],
    nr: usize,
    nc: usize,
) {
    let qk = QK8_0;
    let nb = n / qk;

    assert!(n % qk == 0);
    assert!(nc % NCOLS_INTERLEAVED == 0);

    for x in 0..(nc / NCOLS_INTERLEAVED) {
        let mut sumf = [0.0f32; NCOLS_INTERLEAVED];

        for l in 0..nb {
            let b_block = &vx[x * nb + l];
            let a_block = &vy[l];

            for k in 0..(qk / (2 * BLOCK_SIZE)) {
                for j in 0..NCOLS_INTERLEAVED {
                    let mut sumi = i16x4::splat(0);

                    let idx = k * NCOLS_INTERLEAVED * BLOCK_SIZE + j * BLOCK_SIZE;

                    let v0 = i16x4::from([
                        (b_block.qs[idx] << 4) as i16,
                        (b_block.qs[idx + 1] << 4) as i16,
                        (b_block.qs[idx + 2] << 4) as i16,
                        (b_block.qs[idx + 3] << 4) as i16,
                    ]);
                    let v1 = i16x4::from([
                        (b_block.qs[idx] & 0xF0) as i16,
                        (b_block.qs[idx + 1] & 0xF0) as i16,
                        (b_block.qs[idx + 2] & 0xF0) as i16,
                        (b_block.qs[idx + 3] & 0xF0) as i16,
                    ]);

                    let a_idx0 = k * BLOCK_SIZE;
                    let a_idx1 = k * BLOCK_SIZE + qk / 2;

                    let a0_bytes = [
                        a_block.qs[a_idx0] as i16,
                        a_block.qs[a_idx0 + 1] as i16,
                        a_block.qs[a_idx0 + 2] as i16,
                        a_block.qs[a_idx0 + 3] as i16,
                    ];
                    let a1_bytes = [
                        a_block.qs[a_idx1] as i16,
                        a_block.qs[a_idx1 + 1] as i16,
                        a_block.qs[a_idx1 + 2] as i16,
                        a_block.qs[a_idx1 + 3] as i16,
                    ];

                    let a0 = i16x4::from(a0_bytes);
                    let a1 = i16x4::from(a1_bytes);

                    // Multiply and accumulate
                    sumi += (v0 * a0) >> 4;
                    sumi += (v1 * a1) >> 4;

                    sumf[j] +=
                        sumi.reduce_sum() as f32 * b_block.d[j].to_f32() * a_block.d.to_f32();
                }
            }
        }

        // Store results
        for j in 0..NCOLS_INTERLEAVED {
            s[x * NCOLS_INTERLEAVED + j] = sumf[j];
        }
    }
}

fn gemv_q4_0_4x4_q8_0_simd_parallel(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[BlockQ4_0x4],
    vy: &[BlockQ8_0],
    nr: usize,
    nc: usize,
) {
    let qk = QK8_0;
    let nb = n / qk;

    assert!(n % qk == 0);
    assert!(nc % NCOLS_INTERLEAVED == 0);

    s.par_chunks_mut(NCOLS_INTERLEAVED)
        .enumerate()
        .for_each(|(x, chunk)| {
            let mut sumf = [0.0f32; NCOLS_INTERLEAVED];

            // Process blocks in parallel within each chunk
            (0..nb).for_each(|l| {
                let b_block = &vx[x * nb + l];
                let a_block = &vy[l];

                for k in 0..(qk / (2 * BLOCK_SIZE)) {
                    for j in 0..NCOLS_INTERLEAVED {
                        let mut sumi = i16x4::splat(0);

                        let idx = k * NCOLS_INTERLEAVED * BLOCK_SIZE + j * BLOCK_SIZE;

                        let v0 = i16x4::from([
                            (b_block.qs[idx] << 4) as i16,
                            (b_block.qs[idx + 1] << 4) as i16,
                            (b_block.qs[idx + 2] << 4) as i16,
                            (b_block.qs[idx + 3] << 4) as i16,
                        ]);
                        let v1 = i16x4::from([
                            (b_block.qs[idx] & 0xF0) as i16,
                            (b_block.qs[idx + 1] & 0xF0) as i16,
                            (b_block.qs[idx + 2] & 0xF0) as i16,
                            (b_block.qs[idx + 3] & 0xF0) as i16,
                        ]);

                        let a_idx0 = k * BLOCK_SIZE;
                        let a_idx1 = k * BLOCK_SIZE + qk / 2;

                        let a0_bytes = [
                            a_block.qs[a_idx0] as i16,
                            a_block.qs[a_idx0 + 1] as i16,
                            a_block.qs[a_idx0 + 2] as i16,
                            a_block.qs[a_idx0 + 3] as i16,
                        ];
                        let a1_bytes = [
                            a_block.qs[a_idx1] as i16,
                            a_block.qs[a_idx1 + 1] as i16,
                            a_block.qs[a_idx1 + 2] as i16,
                            a_block.qs[a_idx1 + 3] as i16,
                        ];

                        let a0 = i16x4::from(a0_bytes);
                        let a1 = i16x4::from(a1_bytes);

                        // Multiply and accumulate
                        sumi += (v0 * a0) >> 4;
                        sumi += (v1 * a1) >> 4;

                        sumf[j] +=
                            sumi.reduce_sum() as f32 * b_block.d[j].to_f32() * a_block.d.to_f32();
                    }
                }
            });

            // Store results
            chunk.copy_from_slice(&sumf);
        });
}

// Assembly GEMV function
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn gemv_q4_0_4x4_q8_0_asm(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[BlockQ4_0x4],
    vy: &[BlockQ8_0],
    nr: usize,
    nc: usize,
) {
    // Ensure n is a multiple of QK8_0
    assert!(n % QK8_0 == 0);
    let nb = n / QK8_0;
    let ncols_interleaved = 4;
    let blocklen = 4;

    let b_ptr = vx.as_ptr();
    let a_ptr = vy.as_ptr();
    let res_ptr = s.as_mut_ptr();

    core::arch::asm!(
        // Equivalent assembly code adapted from the original C code
        // Implement the assembly code using Rust's inline assembly
        // Note: Adjustments are made to fit Rust's asm syntax

        // Initialize constants
        "movi v31.16b, #0x4",
        "movi v30.16b, #0xf0",
        "add {b_ptr}, {b_ptr}, #0x8",
        "1:", // Column loop
        "add x22, {a_ptr}, #0x2",
        "movi v29.16b, #0x0",
        "mov x21, {nb}",
        "2:", // Block loop
        "ldr q28, [{b_ptr}, #0x0]",
        "ldr q27, [x22, #0x0]",
        "movi v26.4s, #0x0",
        "sub x20, x22, #0x2",
        "ldr q25, [x22, #0x10]",
        "ldr q24, [{b_ptr}, #0x10]",
        "sub x21, x21, #0x1",
        "add x22, x22, #0x22",
        "ldr q23, [{b_ptr}, #0x20]",
        "ldr q22, [{b_ptr}, #0x30]",
        "ld1r {{ v21.8h }}, [x20]",
        "ldr q20, [{b_ptr}, #-0x8]",
        "sshl v16.16b, v28.16b, v31.16b",
        "and v28.16b, v28.16b, v30.16b",
        "sshl v19.16b, v24.16b, v31.16b",
        "and v24.16b, v24.16b, v30.16b",
        "add {b_ptr}, {b_ptr}, #0x48",
        "sshl v18.16b, v23.16b, v31.16b",
        "and v23.16b, v23.16b, v30.16b",
        ".inst 0x4f9be21a", // sdot v26.4s, v16.16b, v27.4b[0]
        "sshl v17.16b, v22.16b, v31.16b",
        "and v22.16b, v22.16b, v30.16b",
        "fcvtl v21.4s, v21.4h",
        "fcvtl v16.4s, v20.4h",
        ".inst 0x4f99e39a", // sdot v26.4s, v28.16b, v25.4b[0]
        "fmul v16.4s, v16.4s, v21.4s",
        ".inst 0x4fbbe27a", // sdot v26.4s, v19.16b, v27.4b[1]
        ".inst 0x4fb9e31a", // sdot v26.4s, v24.16b, v25.4b[1]
        ".inst 0x4f9bea5a", // sdot v26.4s, v18.16b, v27.4b[2]
        ".inst 0x4f99eafa", // sdot v26.4s, v23.16b, v25.4b[2]
        ".inst 0x4fbbea3a", // sdot v26.4s, v17.16b, v27.4b[3]
        ".inst 0x4fb9eada", // sdot v26.4s, v22.16b, v25.4b[3]
        "scvtf v26.4s, v26.4s, #0x4",
        "fmla v29.4s, v26.4s, v16.4s",
        "cbnz x21, 2b",
        "sub {nc}, {nc}, #0x4",
        "str q29, [{res_ptr}, #0x0]",
        "add {res_ptr}, {res_ptr}, #0x10",
        "cbnz {nc}, 1b",
        b_ptr = inout(reg) b_ptr => _,
        res_ptr = inout(reg) res_ptr => _,
        nc = inout(reg) nc => _,
        a_ptr = in(reg) a_ptr,
        nb = in(reg) nb as isize,
        // Clobbered registers
        out("x20") _, out("x21") _, out("x22") _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        out("v24") _, out("v25") _, out("v26") _, out("v27") _,
        out("v28") _, out("v29") _, out("v30") _, out("v31") _,
        options(nostack)
    );
}

// Unquantized GEMV function
fn gemv_unquantized(n: usize, s: &mut [f32], vx: &[f32], vy: &[f32], nr: usize, nc: usize) {
    for i in 0..nc {
        let mut sum = 0.0f32;
        for k in 0..n {
            sum += vx[i * n + k] * vy[k];
        }
        s[i] = sum;
    }
}

// Unquantized GEMV function using OpenBLAS
fn gemv_unquantized_openblas(
    n: usize,
    s: &mut [f64],
    vx: &[f64],
    vy: &[f64],
    nr: usize,
    nc: usize,
) {
    let alpha = 1.0;
    let beta = 0.0;

    unsafe {
        dgemv(
            b'N',      // trans: 'N' for no transpose
            nc as i32, // m: number of rows
            n as i32,  // n: number of columns
            alpha,     // alpha
            vx,        // matrix A
            nc as i32, // lda: leading dimension of A
            vy,        // vector x
            1,         // incx: increment for x
            beta,      // beta
            s,         // vector y
            1,         // incy: increment for y
        );
    }
}

// Unquantized GEMV function using OpenBLAS
fn gemv_unquantized_blis(n: usize, s: &mut [f32], vx: &[f32], vy: &[f32], nr: usize, nc: usize) {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    unsafe {
        sgemv_(
            &{ b'N' } as *const u8, // trans: 'N' for no transpose
            &(nc as i32),           // m: number of rows
            &(n as i32),            // n: number of columns
            &alpha,                 // alpha
            vx.as_ptr(),            // matrix A
            &(nc as i32),           // lda: leading dimension of A
            vy.as_ptr(),            // vector x
            &1,                     // incx: increment for x
            &beta,                  // beta
            s.as_mut_ptr(),         // vector y
            &1,                     // incy: increment for y
        );
    }
}
// Unquantized GEMV function
fn gemv_unquantized_parallel(
    n: usize,
    s: &mut [f32],
    vx: &[f32],
    vy: &[f32],
    nr: usize,
    nc: usize,
) {
    s.into_par_iter().enumerate().for_each(|(i, s)| {
        let mut sum = 0.0f32;
        for k in 0..n {
            sum += vx[i * n + k] * vy[k];
        }
        *s = sum;
    });
}

// Struct to hold test data
struct TestData {
    vx: Vec<BlockQ4_0x4>,
    vy: Vec<BlockQ8_0>,
    s: Vec<f32>,
    ux: Vec<f32>,
    uy: Vec<f32>,
}

// Function to generate test data
fn generate_test_data(n: usize, nc: usize, random: bool) -> TestData {
    let nb = n / QK8_0;
    let num_vx_blocks = nb * (nc / NCOLS_INTERLEAVED);

    let ux_size = nc * n;
    let uy_size = n;

    let ux: Vec<f32> = (0..ux_size)
        .map(|_| {
            if random {
                rand::random::<f32>() - 0.5
            } else {
                1.0f32
            }
        })
        .collect();

    let uy: Vec<f32> = (0..uy_size)
        .map(|_| {
            if random {
                rand::random::<f32>() - 0.5
            } else {
                1.0f32
            }
        })
        .collect();

    let mut vy = vec![BlockQ8_0::default(); nb];

    quantize_q8_0_4(&uy, &mut vy, n);

    let mut vx = vec![BlockQ4_0x4::default(); num_vx_blocks];

    quantize_q4_0_4x4(&ux, &mut vx, nc, n);

    // Dequantize the first block of vy (A matrix)
    let dequantized_vy_block = dequantize_block_q8_0(&vy[0]);
    let original_ux = &ux[0..QK8_0 * 4];

    // Compare dequantized values with original values
    for (orig, deq) in original_ux.iter().zip(dequantized_vy_block.iter()) {
        assert!(
            (orig - deq).abs() < 1.0,
            "{}",
            format!(
                "Dequantized B matrix values do not match original values: {} != {}",
                orig, deq
            )
        );
    }

    // Dequantize the first block of vx (B matrix)
    let dequantized_vx_blocks = dequantize_block_q4_0x4(&vx[0]);
    let original_uy = &uy[0..QK4_0 * 4];

    // Compare dequantized values with original values
    for (orig, deq) in original_uy.iter().zip(dequantized_vx_blocks.iter()) {
        assert!(
            (orig - deq).abs() < 1.0,
            "{}",
            format!(
                "Dequantized B matrix values do not match original values: {} != {}",
                orig, deq
            )
        );
    }

    let s = vec![0.0f32; nc];

    TestData { vx, vy, s, ux, uy }
}

// Benchmark function for scalar implementation
fn benchmark_scalar_implementation(iterations: usize, n: usize, nc: usize, test_data: &TestData) {
    let name = "Quantized GEMV scalar";

    // Prepare variables
    let mut s_quantized = vec![0.0f32; nc];
    let mut s_unquantized = vec![0.0f32; nc];

    // Benchmark quantized GEMV (scalar)
    let start_q = Instant::now();
    for _ in 0..iterations {
        gemv_q4_0_4x4_q8_0_scalar(
            n,
            &mut s_quantized,
            QK8_0,
            &test_data.vx,
            &test_data.vy,
            1,
            nc,
        );
    }
    let duration_q = start_q.elapsed();

    // Benchmark unquantized GEMV
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemv_unquantized(n, &mut s_unquantized, &test_data.ux, &test_data.uy, 1, nc);
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
    println!("Comparing outputs between quantized and unquantized GEMV:");
    let max_diff = s_quantized
        .iter()
        .zip(s_unquantized.iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .fold(0.0, f64::max);
    println!("Maximum difference: {:.6}", max_diff);

    println!("s_unquantized scalar: {:?}", s_unquantized[0..4].to_vec());
    println!("s_quantized scalar: {:?}", s_quantized[0..4].to_vec());
}

// Benchmark function for assembly implementation
fn benchmark_asm_implementation(iterations: usize, n: usize, nc: usize, test_data: &TestData) {
    let name = "Quantized GEMV assembly";

    // Prepare variables
    let mut s_quantized = vec![0.0f32; nc];
    let mut s_unquantized = vec![0.0f32; nc];

    // Benchmark quantized GEMV (assembly)
    let start_q = Instant::now();
    for _ in 0..iterations {
        unsafe {
            gemv_q4_0_4x4_q8_0_asm(
                n,
                &mut s_quantized,
                QK8_0,
                &test_data.vx,
                &test_data.vy,
                1,
                nc,
            );
        }
    }
    let duration_q = start_q.elapsed();

    // Benchmark unquantized GEMV
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemv_unquantized(n, &mut s_unquantized, &test_data.ux, &test_data.uy, 1, nc);
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
    println!("Comparing outputs between quantized and unquantized GEMV:");
    let max_diff = s_quantized
        .iter()
        .zip(s_unquantized.iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .fold(0.0, f64::max);
    println!("Maximum difference: {:.6}", max_diff);

    println!("s_unquantized asm: {:?}", s_unquantized[0..4].to_vec());
    println!("s_quantized asm: {:?}", s_quantized[0..4].to_vec());
}

// Benchmark function for simd implementation
fn benchmark_simd_implementation(iterations: usize, n: usize, nc: usize, test_data: &TestData) {
    let name = "Quantized GEMV Simd";

    // Prepare variables
    let mut s_quantized = vec![0.0f32; nc];
    let mut s_unquantized = vec![0.0f32; nc];

    // Benchmark quantized GEMV (assembly)
    let start_q = Instant::now();
    for _ in 0..iterations {
        gemv_q4_0_4x4_q8_0_simd(
            n,
            &mut s_quantized,
            QK8_0,
            &test_data.vx,
            &test_data.vy,
            1,
            nc,
        );
    }
    let duration_q = start_q.elapsed();

    // Benchmark unquantized GEMV
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemv_unquantized(n, &mut s_unquantized, &test_data.ux, &test_data.uy, 1, nc);
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
    println!("Comparing outputs between quantized and unquantized GEMV:");
    let max_diff = s_quantized
        .iter()
        .zip(s_unquantized.iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .fold(0.0, f64::max);
    println!("Maximum difference: {:.6}", max_diff);

    println!("s_unquantized simd: {:?}", s_unquantized[0..4].to_vec());
    println!("s_quantized simd: {:?}", s_quantized[0..4].to_vec());
}

// Benchmark function for OpenBLAS implementation
fn benchmark_openblas_implementation(iterations: usize, n: usize, nc: usize, test_data: &TestData) {
    let name = "Unquantized GEMV OpenBLAS";

    // Prepare variables
    let mut s_unquantized = vec![0.0f32; nc];

    let mut s_unquantized_openblas = vec![0.0f64; nc];

    let ux_openblas: Vec<f64> = test_data.ux.iter().map(|&x| x as f64).collect();
    let uy_openblas: Vec<f64> = test_data.uy.iter().map(|&x| x as f64).collect();

    // Benchmark unquantized GEMV (OpenBLAS)
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemv_unquantized_openblas(
            n,
            &mut s_unquantized_openblas,
            &ux_openblas,
            &uy_openblas,
            1,
            nc,
        );
    }
    let duration_uq_openblas = start_uq.elapsed();

    // Benchmark unquantized GEMV
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemv_unquantized(n, &mut s_unquantized, &test_data.ux, &test_data.uy, 1, nc);
    }
    let duration_uq = start_uq.elapsed();

    println!();

    println!(
        "{} (OpenBLAS): {:.2} ms per iteration",
        name,
        (duration_uq_openblas.as_secs_f64() * 1000.0) / iterations as f64
    );

    println!(
        "Unquantized GEMV (Unquantized): {:.2} ms per iteration",
        (duration_uq.as_secs_f64() * 1000.0) / iterations as f64
    );

    println!("s_unquantized openblas: {:?}", s_unquantized[0..4].to_vec());

    println!("Comparing outputs between openblas and unquantized GEMV:");
    let max_diff = s_unquantized_openblas
        .iter()
        .zip(s_unquantized.iter())
        .map(|(&a, &b)| (a as f32 - b).abs() as f64)
        .fold(0.0, f64::max);
    println!("Maximum difference: {:.6}", max_diff);

    println!(
        "s_unquantized openblas: {:?}",
        s_unquantized_openblas[0..4].to_vec()
    );
    println!("s_unquantized normal: {:?}", s_unquantized[0..4].to_vec());
}

// Benchmark function for OpenBLAS implementation
fn benchmark_blis_implementation(iterations: usize, n: usize, nc: usize, test_data: &TestData) {
    let name = "Unquantized GEMV BLIS";

    // Prepare variables
    let mut s_unquantized = vec![0.0f32; nc];

    let mut s_unquantized_blis = vec![0.0f32; nc];

    // Benchmark unquantized GEMV (OpenBLAS)
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemv_unquantized_blis(
            n,
            &mut s_unquantized_blis,
            &test_data.ux,
            &test_data.uy,
            1,
            nc,
        );
    }
    let duration_uq_openblas = start_uq.elapsed();

    // Benchmark unquantized GEMV
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemv_unquantized(n, &mut s_unquantized, &test_data.ux, &test_data.uy, 1, nc);
    }
    let duration_uq = start_uq.elapsed();

    println!();

    println!(
        "{} (blis): {:.2} ms per iteration",
        name,
        (duration_uq_openblas.as_secs_f64() * 1000.0) / iterations as f64
    );

    println!(
        "Unquantized GEMV (Unquantized): {:.2} ms per iteration",
        (duration_uq.as_secs_f64() * 1000.0) / iterations as f64
    );

    println!("s_unquantized openblas: {:?}", s_unquantized[0..4].to_vec());

    println!("Comparing outputs between blis and unquantized GEMV:");
    let max_diff = s_unquantized_blis
        .iter()
        .zip(s_unquantized.iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .fold(0.0, f64::max);
    println!("Maximum difference: {:.6}", max_diff);

    println!(
        "s_unquantized blis: {:?}",
        s_unquantized_blis[0..4].to_vec()
    );
    println!("s_unquantized normal: {:?}", s_unquantized[0..4].to_vec());
}

fn main() {
    let n = 128;
    let nc = 128;
    let iterations = 1;

    println!(
        "Benchmarking with n={}, nc={}, iterations={}",
        n, nc, iterations
    );

    // Generate test data
    let test_data = generate_test_data(n, nc, true);

    // Benchmark scalar implementation
    benchmark_scalar_implementation(iterations, n, nc, &test_data);

    // Benchmark assembly implementation
    benchmark_asm_implementation(iterations, n, nc, &test_data);

    // Benchmark simd implementation
    benchmark_simd_implementation(iterations, n, nc, &test_data);

    // Benchmark OpenBLAS implementation
    benchmark_openblas_implementation(iterations, n, nc, &test_data);

    // Benchmark BLIS implementation
    benchmark_blis_implementation(iterations, n, nc, &test_data);
}
