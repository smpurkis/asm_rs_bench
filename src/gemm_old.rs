#![feature(portable_simd)] // Enable std::arch module for intrinsics

extern crate blas_sys;
extern crate blis_src;

mod quantization;

use blas::dgemm;
use blas_sys::sgemm_;
use half::f16;
use num::SimdInt;
use quantization::*;
use rand::prelude::*;
// use wide::i32x4;
use rayon::prelude::*;
use std::simd::*;
use std::time::Instant;

fn gemm_q4_0_4x4_q8_0_scalar(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[BlockQ4_0x4],
    vy: &[BlockQ8_0x4],
    nr: usize,
    nc: usize,
) {
    let qk = QK8_0;
    let nb = n / qk;
    let blocklen = 4;

    assert!(n % qk == 0);
    assert!(nr % 4 == 0);
    assert!(nc % NCOLS_INTERLEAVED == 0);

    for y in 0..(nr / 4) {
        let a_ptr = &vy[y * nb..];

        for x in 0..(nc / NCOLS_INTERLEAVED) {
            let b_ptr = &vx[x * nb..];

            let mut sumf = [[0.0f32; NCOLS_INTERLEAVED]; 4];

            for l in 0..nb {
                let a_block = &a_ptr[l];
                let b_block = &b_ptr[l];

                for k in 0..(qk / (2 * blocklen)) {
                    for m in 0..4 {
                        for j in 0..NCOLS_INTERLEAVED {
                            let mut sumi = 0i32;

                            for i in 0..blocklen {
                                let idx_b = k * NCOLS_INTERLEAVED * blocklen + j * blocklen + i;
                                let byte = b_block.qs[idx_b];

                                let v0 = (byte << 4) as i32;
                                let v1 = (byte & 0xF0) as i32;

                                let idx_a0 = k * 4 * blocklen + m * blocklen + i;
                                let idx_a1 = idx_a0 + qk / 2 * 4;

                                let a0 = a_block.qs[idx_a0];
                                let a1 = a_block.qs[idx_a1];

                                sumi += ((v0 * a0 as i32) >> 4) + ((v1 * a1 as i32) >> 4);
                            }
                            let b_scale = b_block.d[j].to_f32();
                            let a_scale = a_block.d[m].to_f32();

                            sumf[m][j] += (sumi as f32) * b_scale * a_scale;
                        }
                    }
                }
            }

            // Store results
            for m in 0..4 {
                for j in 0..NCOLS_INTERLEAVED {
                    let s_index = ((y * 4 + m) * nc) + (x * NCOLS_INTERLEAVED) + j;
                    s[s_index] = sumf[m][j];
                }
            }
        }
    }
}

// Parallel GEMV function for quantized data using Rayon
fn gemm_q4_0_4x4_q8_0_scalar_parallel(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[BlockQ4_0x4],
    vy: &[BlockQ8_0x4],
    nr: usize,
    nc: usize,
) {
    let qk = QK8_0;
    let nb = n / qk;
    let blocklen = 4;

    assert!(n % qk == 0);
    assert!(nr % 4 == 0);
    assert!(nc % NCOLS_INTERLEAVED == 0);

    for y in 0..(nr / 4) {
        let a_ptr = &vy[y * nb..];

        // for x in 0..(nc / NCOLS_INTERLEAVED) {
        let sumfs = (0..(nc / NCOLS_INTERLEAVED))
            .into_par_iter()
            .map(|x| {
                let b_ptr = &vx[x * nb..];

                let mut sumf = [[0.0f32; NCOLS_INTERLEAVED]; 4];

                for l in 0..nb {
                    let a_block = &a_ptr[l];
                    let b_block = &b_ptr[l];

                    for k in 0..(qk / (2 * blocklen)) {
                        for m in 0..4 {
                            for j in 0..NCOLS_INTERLEAVED {
                                let mut sumi = 0i32;

                                for i in 0..blocklen {
                                    let idx_b = k * NCOLS_INTERLEAVED * blocklen + j * blocklen + i;
                                    let byte = b_block.qs[idx_b];

                                    let v0 = (byte << 4) as i32;
                                    let v1 = (byte & 0xF0) as i32;

                                    let idx_a0 = k * 4 * blocklen + m * blocklen + i;
                                    let idx_a1 = idx_a0 + qk / 2 * 4;

                                    let a0 = a_block.qs[idx_a0];
                                    let a1 = a_block.qs[idx_a1];

                                    sumi += ((v0 * a0 as i32) >> 4) + ((v1 * a1 as i32) >> 4);
                                }
                                let b_scale = b_block.d[j].to_f32();
                                let a_scale = a_block.d[m].to_f32();

                                sumf[m][j] += (sumi as f32) * b_scale * a_scale;
                            }
                        }
                    }
                }
                (x, sumf)
            })
            .collect::<Vec<_>>();

        // Store results
        for (x, sumf) in sumfs.into_iter() {
            for m in 0..4 {
                for j in 0..NCOLS_INTERLEAVED {
                    let s_index = ((y * 4 + m) * nc) + (x * NCOLS_INTERLEAVED) + j;
                    s[s_index] = sumf[m][j];
                }
            }
        }
    }
}

fn gemm_q4_0_q8_0_scalar(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[BlockQ4_0],
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
                let block_x = &vx[ir * nb + ib];
                let block_y = &vy[ic * nb + ib];

                // Get scales for both blocks
                let scale_x = block_x.d.to_f32();
                let scale_y = block_y.d.to_f32();

                // Process all values in the block
                for i in 0..QK4_0 / 2 {
                    // Extract two 4-bit values from each byte
                    let x_byte = block_x.qs[i];
                    let x_val0 = (x_byte & 0xF) as i8 - 8;
                    let x_val1 = ((x_byte >> 4) & 0xF) as i8 - 8;

                    // Get corresponding 8-bit values
                    let y_val0 = block_y.qs[i * 2];
                    let y_val1 = block_y.qs[i * 2 + 1];

                    // Multiply values and accumulate with scaling
                    sum += scale_x
                        * scale_y
                        * ((x_val0 as f32) * (y_val0 as f32) + (x_val1 as f32) * (y_val1 as f32));
                }
            }

            // Store result
            s[ir * bs + ic] = sum;
        }
    }
}

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
                let block_x = &vx[ir * nb + ib];
                let block_y = &vy[ic * nb + ib];

                // Get scales for both blocks
                let scale_x = block_x.d.to_f32();
                let scale_y = block_y.d.to_f32();

                // Process all values in the block
                for i in 0..QK8_0 {
                    let x_val = block_x.qs[i];
                    let y_val = block_y.qs[i];

                    // Multiply values and accumulate with scaling
                    sum += scale_x * scale_y * (x_val as f32) * (y_val as f32);
                }
            }

            // Store result
            s[ir * bs + ic] = sum;
        }
    }
}

fn gemm_q4_0_4x4_q8_0_simd(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[BlockQ4_0x4],
    vy: &[BlockQ8_0x4],
    nr: usize,
    nc: usize,
) {
}

fn gemm_q4_0_4x4_q8_0_simd_parallel(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[BlockQ4_0x4],
    vy: &[BlockQ8_0x4],
    nr: usize,
    nc: usize,
) {
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn gemm_q4_0_4x4_q8_0_asm(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[BlockQ4_0x4],
    vy: &[BlockQ8_0x4],
    nr: usize,
    nc: usize,
) {
    let qk = QK8_0;
    let nb = n / qk;
    let res_stride = bs * std::mem::size_of::<f32>();
    println!("res_stride: {}", res_stride);

    assert!(n % qk == 0);
    assert!(nr % 4 == 0);
    assert!(nc % NCOLS_INTERLEAVED == 0);

    let b_ptr = vx.as_ptr();
    let a_ptr = vy.as_ptr();
    let res_ptr = s.as_mut_ptr();

    core::arch::asm!(
        // Assembly code adapted from the C code
        "mov x10, {nr}",
        "mov x9, #0x88",
        "cmp x10, #0x10",
        "mul x9, {nb}, x9",
        "blt 4f",
        "1:",  // Row loop
        "add x28, {b_ptr}, #0x8",
        "mov x27, {nc}",
        "lsl x20, {res_stride}, #4",
        "add x26, {res_ptr}, x20",
        "2:",  // Column loop
        "add x25, {a_ptr}, #0x8",
        "movi v15.16b, #0x0",
        "movi v19.16b, #0x0",
        "mov x24, {nb}",
        "add x23, x25, x9",
        "movi v18.16b, #0x0",
        "movi v14.16b, #0x0",
        "add x22, x23, x9",
        "movi v11.16b, #0x0",
        "movi v13.16b, #0x0",
        "add x21, x22, x9",
        "movi v23.16b, #0x0",
        "movi v16.16b, #0x0",
        "movi v25.16b, #0x0",
        "movi v7.16b, #0x0",
        "movi v0.16b, #0x0",
        "movi v4.16b, #0x0",
        "movi v5.16b, #0x0",
        "movi v21.16b, #0x0",
        "movi v8.16b, #0x0",
        "movi v1.16b, #0x0",
        "3:",  // Block loop
        "ldr q3, [x28, #0x0]",
        "ldr q31, [x25, #0x0]",
        "movi v28.16b, #0x4",
        "movi v10.4s, #0x0",
        "ldr q22, [x28, #0x10]",
        "ldr q6, [x25, #0x10]",
        "movi v29.4s, #0x0",
        "movi v9.4s, #0x0",
        "ldr q27, [x28, #0x20]",
        "ldr q30, [x28, #0x30]",
        "movi v20.4s, #0x0",
        "movi v24.16b, #0xf0",
        "ldr d2, [x25, #-0x8]",
        "ldr d26, [x23, #-0x8]",
        "sshl v12.16b, v3.16b, v28.16b",
        "sub x20, x28, #0x8",
        "ldr d17, [x20, #0x0]",
        "and v3.16b, v3.16b, v24.16b",
        "subs x24, x24, #0x1",
        "add x28, x28, #0x48",
        ".inst 0x4f9fe18a  // sdot v10.4s, v12.16b, v31.4b[0]",
        ".inst 0x4fbfe19d  // sdot v29.4s, v12.16b, v31.4b[1]",
        ".inst 0x4f9fe989  // sdot v9.4s, v12.16b, v31.4b[2]",
        ".inst 0x4fbfe994  // sdot v20.4s, v12.16b, v31.4b[3]",
        "sshl v31.16b, v22.16b, v28.16b",
        "and v22.16b, v22.16b, v24.16b",
        "fcvtl v17.4s, v17.4h",
        "fcvtl v2.4s, v2.4h",
        "fcvtl v26.4s, v26.4h",
        ".inst 0x4f86e3ea  // sdot v10.4s, v31.16b, v6.4b[0]",
        ".inst 0x4fa6e3fd  // sdot v29.4s, v31.16b, v6.4b[1]",
        ".inst 0x4f86ebe9  // sdot v9.4s, v31.16b, v6.4b[2]",
        ".inst 0x4fa6ebf4  // sdot v20.4s, v31.16b, v6.4b[3]",
        "sshl v6.16b, v27.16b, v28.16b",
        "sshl v28.16b, v30.16b, v28.16b",
        "and v27.16b, v27.16b, v24.16b",
        "and v30.16b, v30.16b, v24.16b",
        "ldr q24, [x25, #0x20]",
        ".inst 0x4f98e0ca  // sdot v10.4s, v6.16b, v24.4b[0]",
        ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]",
        ".inst 0x4f98e8c9  // sdot v9.4s, v6.16b, v24.4b[2]",
        ".inst 0x4fb8e8d4  // sdot v20.4s, v6.16b, v24.4b[3]",
        "ldr q24, [x25, #0x30]",
        ".inst 0x4f98e38a  // sdot v10.4s, v28.16b, v24.4b[0]",
        ".inst 0x4fb8e39d  // sdot v29.4s, v28.16b, v24.4b[1]",
        ".inst 0x4f98eb89  // sdot v9.4s, v28.16b, v24.4b[2]",
        ".inst 0x4fb8eb94  // sdot v20.4s, v28.16b, v24.4b[3]",
        "ldr q24, [x25, #0x40]",
        ".inst 0x4f98e06a  // sdot v10.4s, v3.16b, v24.4b[0]",
        ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]",
        ".inst 0x4f98e869  // sdot v9.4s, v3.16b, v24.4b[2]",
        ".inst 0x4fb8e874  // sdot v20.4s, v3.16b, v24.4b[3]",
        "ldr q24, [x25, #0x50]",
        ".inst 0x4f98e2ca  // sdot v10.4s, v22.16b, v24.4b[0]",
        ".inst 0x4fb8e2dd  // sdot v29.4s, v22.16b, v24.4b[1]",
        ".inst 0x4f98eac9  // sdot v9.4s, v22.16b, v24.4b[2]",
        ".inst 0x4fb8ead4  // sdot v20.4s, v22.16b, v24.4b[3]",
        "ldr q24, [x25, #0x60]",
        ".inst 0x4f98e36a  // sdot v10.4s, v27.16b, v24.4b[0]",
        ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]",
        ".inst 0x4f98eb69  // sdot v9.4s, v27.16b, v24.4b[2]",
        ".inst 0x4fb8eb74  // sdot v20.4s, v27.16b, v24.4b[3]",
        "ldr q24, [x25, #0x70]",
        "add x25, x25, #0x88",
        ".inst 0x4f98e3ca  // sdot v10.4s, v30.16b, v24.4b[0]",
        ".inst 0x4fb8e3dd  // sdot v29.4s, v30.16b, v24.4b[1]",
        ".inst 0x4f98ebc9  // sdot v9.4s, v30.16b, v24.4b[2]",
        ".inst 0x4fb8ebd4  // sdot v20.4s, v30.16b, v24.4b[3]",
        "fmul v24.4s, v17.4s, v2.s[0]",
        "scvtf v10.4s, v10.4s, #0x4",
        "scvtf v29.4s, v29.4s, #0x4",
        "scvtf v9.4s, v9.4s, #0x4",
        "scvtf v20.4s, v20.4s, #0x4",
        "fmla v15.4s, v10.4s, v24.4s",
        "ldr q24, [x23, #0x0]",
        "fmul v10.4s, v17.4s, v2.s[1]",
        "fmla v19.4s, v29.4s, v10.4s",
        "ldr q10, [x23, #0x10]",
        "fmul v29.4s, v17.4s, v2.s[2]",
        "fmul v2.4s, v17.4s, v2.s[3]",
        "fmla v18.4s, v9.4s, v29.4s",
        "movi v9.4s, #0x0",
        "movi v29.4s, #0x0",
        ".inst 0x4f98e189  // sdot v9.4s, v12.16b, v24.4b[0]",
        ".inst 0x4fb8e19d  // sdot v29.4s, v12.16b, v24.4b[1]",
        "fmla v14.4s, v20.4s, v2.4s",
        "movi v20.4s, #0x0",
        "movi v2.4s, #0x0",
        ".inst 0x4f98e994  // sdot v20.4s, v12.16b, v24.4b[2]",
        ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]",
        "ldr q24, [x23, #0x20]",
        ".inst 0x4f8ae3e9  // sdot v9.4s, v31.16b, v10.4b[0]",
        ".inst 0x4faae3fd  // sdot v29.4s, v31.16b, v10.4b[1]",
        ".inst 0x4f8aebf4  // sdot v20.4s, v31.16b, v10.4b[2]",
        ".inst 0x4faaebe2  // sdot v2.4s, v31.16b, v10.4b[3]",
        "ldr q10, [x23, #0x30]",
        ".inst 0x4f98e0c9  // sdot v9.4s, v6.16b, v24.4b[0]",
        ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]",
        ".inst 0x4f98e8d4  // sdot v20.4s, v6.16b, v24.4b[2]",
        ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]",
        "ldr q24, [x23, #0x40]",
        ".inst 0x4f8ae389  // sdot v9.4s, v28.16b, v10.4b[0]",
        ".inst 0x4faae39d  // sdot v29.4s, v28.16b, v10.4b[1]",
        ".inst 0x4f8aeb94  // sdot v20.4s, v28.16b, v10.4b[2]",
        ".inst 0x4faaeb82  // sdot v2.4s, v28.16b, v10.4b[3]",
        "ldr q10, [x23, #0x50]",
        ".inst 0x4f98e069  // sdot v9.4s, v3.16b, v24.4b[0]",
        ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]",
        ".inst 0x4f98e874  // sdot v20.4s, v3.16b, v24.4b[2]",
        ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]",
        "ldr q24, [x23, #0x60]",
        ".inst 0x4f8ae2c9  // sdot v9.4s, v22.16b, v10.4b[0]",
        ".inst 0x4faae2dd  // sdot v29.4s, v22.16b, v10.4b[1]",
        ".inst 0x4f8aead4  // sdot v20.4s, v22.16b, v10.4b[2]",
        ".inst 0x4faaeac2  // sdot v2.4s, v22.16b, v10.4b[3]",
        "ldr q10, [x23, #0x70]",
        "add x23, x23, #0x88",
        ".inst 0x4f98e369  // sdot v9.4s, v27.16b, v24.4b[0]",
        ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]",
        ".inst 0x4f98eb74  // sdot v20.4s, v27.16b, v24.4b[2]",
        ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]",
        "ldr q24, [x22, #0x0]",
        ".inst 0x4f8ae3c9  // sdot v9.4s, v30.16b, v10.4b[0]",
        ".inst 0x4faae3dd  // sdot v29.4s, v30.16b, v10.4b[1]",
        ".inst 0x4f8aebd4  // sdot v20.4s, v30.16b, v10.4b[2]",
        ".inst 0x4faaebc2  // sdot v2.4s, v30.16b, v10.4b[3]",
        "fmul v10.4s, v17.4s, v26.s[0]",
        "scvtf v9.4s, v9.4s, #0x4",
        "scvtf v29.4s, v29.4s, #0x4",
        "scvtf v20.4s, v20.4s, #0x4",
        "scvtf v2.4s, v2.4s, #0x4",
        "fmla v11.4s, v9.4s, v10.4s",
        "ldr q9, [x22, #0x10]",
        "fmul v10.4s, v17.4s, v26.s[1]",
        "fmla v13.4s, v29.4s, v10.4s",
        "ldr d29, [x22, #-0x8]",
        "fmul v10.4s, v17.4s, v26.s[2]",
        "fmul v26.4s, v17.4s, v26.s[3]",
        "fcvtl v29.4s, v29.4h",
        "fmla v23.4s, v20.4s, v10.4s",
        "movi v20.4s, #0x0",
        "movi v10.4s, #0x0",
        "fmla v16.4s, v2.4s, v26.4s",
        "movi v26.4s, #0x0",
        "movi v2.4s, #0x0",
        ".inst 0x4f98e194  // sdot v20.4s, v12.16b, v24.4b[0]",
        ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]",
        ".inst 0x4f98e99a  // sdot v26.4s, v12.16b, v24.4b[2]",
        ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]",
        "ldr q24, [x22, #0x20]",
        ".inst 0x4f89e3f4  // sdot v20.4s, v31.16b, v9.4b[0]",
        ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]",
        ".inst 0x4f89ebfa  // sdot v26.4s, v31.16b, v9.4b[2]",
        ".inst 0x4fa9ebe2  // sdot v2.4s, v31.16b, v9.4b[3]",
        "ldr q9, [x22, #0x30]",
        ".inst 0x4f98e0d4  // sdot v20.4s, v6.16b, v24.4b[0]",
        ".inst 0x4fb8e0ca  // sdot v10.4s, v6.16b, v24.4b[1]",
        ".inst 0x4f98e8da  // sdot v26.4s, v6.16b, v24.4b[2]",
        ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]",
        "ldr q24, [x22, #0x40]",
        ".inst 0x4f89e394  // sdot v20.4s, v28.16b, v9.4b[0]",
        ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]",
        ".inst 0x4f89eb9a  // sdot v26.4s, v28.16b, v9.4b[2]",
        ".inst 0x4fa9eb82  // sdot v2.4s, v28.16b, v9.4b[3]",
        "ldr q9, [x22, #0x50]",
        ".inst 0x4f98e074  // sdot v20.4s, v3.16b, v24.4b[0]",
        ".inst 0x4fb8e06a  // sdot v10.4s, v3.16b, v24.4b[1]",
        ".inst 0x4f98e87a  // sdot v26.4s, v3.16b, v24.4b[2]",
        ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]",
        "ldr q24, [x22, #0x60]",
        ".inst 0x4f89e2d4  // sdot v20.4s, v22.16b, v9.4b[0]",
        ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]",
        ".inst 0x4f89eada  // sdot v26.4s, v22.16b, v9.4b[2]",
        ".inst 0x4fa9eac2  // sdot v2.4s, v22.16b, v9.4b[3]",
        "ldr q9, [x22, #0x70]",
        "add x22, x22, #0x88",
        ".inst 0x4f98e374  // sdot v20.4s, v27.16b, v24.4b[0]",
        ".inst 0x4fb8e36a  // sdot v10.4s, v27.16b, v24.4b[1]",
        ".inst 0x4f98eb7a  // sdot v26.4s, v27.16b, v24.4b[2]",
        ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]",
        "ldr q24, [x21, #0x0]",
        ".inst 0x4f89e3d4  // sdot v20.4s, v30.16b, v9.4b[0]",
        ".inst 0x4fa9e3ca  // sdot v10.4s, v30.16b, v9.4b[1]",
        ".inst 0x4f89ebda  // sdot v26.4s, v30.16b, v9.4b[2]",
        ".inst 0x4fa9ebc2  // sdot v2.4s, v30.16b, v9.4b[3]",
        "fmul v9.4s, v17.4s, v29.s[0]",
        "scvtf v20.4s, v20.4s, #0x4",
        "scvtf v10.4s, v10.4s, #0x4",
        "scvtf v26.4s, v26.4s, #0x4",
        "scvtf v2.4s, v2.4s, #0x4",
        "fmla v25.4s, v20.4s, v9.4s",
        "ldr q9, [x21, #0x10]",
        "fmul v20.4s, v17.4s, v29.s[1]",
        "fmla v7.4s, v10.4s, v20.4s",
        "ldr d20, [x21, #-0x8]",
        "fmul v10.4s, v17.4s, v29.s[2]",
        "fmul v29.4s, v17.4s, v29.s[3]",
        "fcvtl v20.4s, v20.4h",
        "fmla v0.4s, v26.4s, v10.4s",
        "movi v26.4s, #0x0",
        "movi v10.4s, #0x0",
        "fmla v4.4s, v2.4s, v29.4s",
        "movi v2.4s, #0x0",
        "movi v29.4s, #0x0",
        ".inst 0x4f98e19a  // sdot v26.4s, v12.16b, v24.4b[0]",
        ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]",
        ".inst 0x4f98e982  // sdot v2.4s, v12.16b, v24.4b[2]",
        ".inst 0x4fb8e99d  // sdot v29.4s, v12.16b, v24.4b[3]",
        "ldr q12, [x21, #0x20]",
        "fmul v24.4s, v17.4s, v20.s[0]",
        ".inst 0x4f89e3fa  // sdot v26.4s, v31.16b, v9.4b[0]",
        ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]",
        ".inst 0x4f89ebe2  // sdot v2.4s, v31.16b, v9.4b[2]",
        ".inst 0x4fa9ebfd  // sdot v29.4s, v31.16b, v9.4b[3]",
        "ldr q9, [x21, #0x30]",
        "fmul v31.4s, v17.4s, v20.s[1]",
        ".inst 0x4f8ce0da  // sdot v26.4s, v6.16b, v12.4b[0]",
        ".inst 0x4face0ca  // sdot v10.4s, v6.16b, v12.4b[1]",
        ".inst 0x4f8ce8c2  // sdot v2.4s, v6.16b, v12.4b[2]",
        ".inst 0x4face8dd  // sdot v29.4s, v6.16b, v12.4b[3]",
        "ldr q12, [x21, #0x40]",
        "fmul v6.4s, v17.4s, v20.s[2]",
        "fmul v20.4s, v17.4s, v20.s[3]",
        ".inst 0x4f89e39a  // sdot v26.4s, v28.16b, v9.4b[0]",
        ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]",
        ".inst 0x4f89eb82  // sdot v2.4s, v28.16b, v9.4b[2]",
        ".inst 0x4fa9eb9d  // sdot v29.4s, v28.16b, v9.4b[3]",
        "ldr q9, [x21, #0x50]",
        ".inst 0x4f8ce07a  // sdot v26.4s, v3.16b, v12.4b[0]",
        ".inst 0x4face06a  // sdot v10.4s, v3.16b, v12.4b[1]",
        ".inst 0x4f8ce862  // sdot v2.4s, v3.16b, v12.4b[2]",
        ".inst 0x4face87d  // sdot v29.4s, v3.16b, v12.4b[3]",
        "ldr q12, [x21, #0x60]",
        ".inst 0x4f89e2da  // sdot v26.4s, v22.16b, v9.4b[0]",
        ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]",
        ".inst 0x4f89eac2  // sdot v2.4s, v22.16b, v9.4b[2]",
        ".inst 0x4fa9eadd  // sdot v29.4s, v22.16b, v9.4b[3]",
        "ldr q17, [x21, #0x70]",
        "add x21, x21, #0x88",
        ".inst 0x4f8ce37a  // sdot v26.4s, v27.16b, v12.4b[0]",
        ".inst 0x4face36a  // sdot v10.4s, v27.16b, v12.4b[1]",
        ".inst 0x4f8ceb62  // sdot v2.4s, v27.16b, v12.4b[2]",
        ".inst 0x4faceb7d  // sdot v29.4s, v27.16b, v12.4b[3]",
        ".inst 0x4f91e3da  // sdot v26.4s, v30.16b, v17.4b[0]",
        ".inst 0x4fb1e3ca  // sdot v10.4s, v30.16b, v17.4b[1]",
        ".inst 0x4f91ebc2  // sdot v2.4s, v30.16b, v17.4b[2]",
        ".inst 0x4fb1ebdd  // sdot v29.4s, v30.16b, v17.4b[3]",
        "scvtf v26.4s, v26.4s, #0x4",
        "scvtf v10.4s, v10.4s, #0x4",
        "fmla v5.4s, v26.4s, v24.4s",
        "scvtf v2.4s, v2.4s, #0x4",
        "scvtf v29.4s, v29.4s, #0x4",
        "fmla v21.4s, v10.4s, v31.4s",
        "fmla v8.4s, v2.4s, v6.4s",
        "fmla v1.4s, v29.4s, v20.4s",
        "bgt 3b",
        "mov x20, {res_ptr}",
        "subs x27, x27, #0x4",
        "add {res_ptr}, {res_ptr}, #0x10",
        "str q15, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q19, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q18, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q14, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q11, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q13, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q23, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q16, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q25, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q7, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q0, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q4, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q5, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q21, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q8, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "str q1, [x20, #0x0]",
        "bne 2b",
        "mov x20, #0x4",
        "sub x10, x10, #0x10",
        "cmp x10, #0x10",
        "mov {res_ptr}, x26",
        "madd {a_ptr}, x20, x9, {a_ptr}",
        "bge 1b",
        "4:",  // Row loop skip
        "cbz x10, 9f",
        "5:",  // Row tail: Row loop
        "add x24, {b_ptr}, #0x8",
        "mov x23, {nc}",
        "add x22, {res_ptr}, {res_stride}, LSL #2",
        "6:",  // Row tail: Column loop
        "movi v15.16b, #0x0",
        "movi v19.16b, #0x0",
        "add x25, {a_ptr}, #0x8",
        "mov x21, {nb}",
        "movi v18.16b, #0x0",
        "movi v14.16b, #0x0",
        "7:",  // Row tail: Block loop
        "ldr q7, [x24, #0x0]",
        "ldr q5, [x25, #0x0]",
        "movi v9.16b, #0x4",
        "movi v4.4s, #0x0",
        "ldr q3, [x24, #0x10]",
        "ldr q2, [x25, #0x10]",
        "movi v1.4s, #0x0",
        "movi v0.4s, #0x0",
        "ldr q13, [x24, #0x20]",
        "ldr q31, [x25, #0x20]",
        "movi v30.4s, #0x0",
        "movi v29.16b, #0xf0",
        "ldr q28, [x24, #0x30]",
        "ldr q27, [x25, #0x30]",
        "sshl v20.16b, v7.16b, v9.16b",
        "sub x20, x24, #0x8",
        "ldr q26, [x25, #0x40]",
        "ldr q25, [x25, #0x50]",
        "sshl v17.16b, v3.16b, v9.16b",
        "and v7.16b, v7.16b, v29.16b",
        "ldr q24, [x25, #0x60]",
        "ldr q16, [x25, #0x70]",
        "sshl v22.16b, v13.16b, v9.16b",
        "and v3.16b, v3.16b, v29.16b",
        "ldr d21, [x20, #0x0]",
        "ldr d12, [x25, #-0x8]",
        ".inst 0x4f85e284  // sdot v4.4s, v20.16b, v5.4b[0]",
        ".inst 0x4fa5e281  // sdot v1.4s, v20.16b, v5.4b[1]",
        ".inst 0x4f85ea80  // sdot v0.4s, v20.16b, v5.4b[2]",
        ".inst 0x4fa5ea9e  // sdot v30.4s, v20.16b, v5.4b[3]",
        "sshl v9.16b, v28.16b, v9.16b",
        "subs x21, x21, #0x1",
        "and v13.16b, v13.16b, v29.16b",
        "and v28.16b, v28.16b, v29.16b",
        "add x25, x25, #0x88",
        "add x24, x24, #0x48",
        "fcvtl v21.4s, v21.4h",
        "fcvtl v12.4s, v12.4h",
        ".inst 0x4f82e224  // sdot v4.4s, v17.16b, v2.4b[0]",
        ".inst 0x4fa2e221  // sdot v1.4s, v17.16b, v2.4b[1]",
        ".inst 0x4f82ea20  // sdot v0.4s, v17.16b, v2.4b[2]",
        ".inst 0x4fa2ea3e  // sdot v30.4s, v17.16b, v2.4b[3]",
        "fmul v11.4s, v21.4s, v12.s[0]",
        "fmul v23.4s, v21.4s, v12.s[1]",
        "fmul v17.4s, v21.4s, v12.s[2]",
        ".inst 0x4f9fe2c4  // sdot v4.4s, v22.16b, v31.4b[0]",
        "fmul v6.4s, v21.4s, v12.s[3]",
        ".inst 0x4fbfe2c1  // sdot v1.4s, v22.16b, v31.4b[1]",
        ".inst 0x4f9feac0  // sdot v0.4s, v22.16b, v31.4b[2]",
        ".inst 0x4fbfeade  // sdot v30.4s, v22.16b, v31.4b[3]",
        ".inst 0x4f9be124  // sdot v4.4s, v9.16b, v27.4b[0]",
        ".inst 0x4fbbe121  // sdot v1.4s, v9.16b, v27.4b[1]",
        ".inst 0x4f9be920  // sdot v0.4s, v9.16b, v27.4b[2]",
        ".inst 0x4fbbe93e  // sdot v30.4s, v9.16b, v27.4b[3]",
        ".inst 0x4f9ae0e4  // sdot v4.4s, v7.16b, v26.4b[0]",
        ".inst 0x4fbae0e1  // sdot v1.4s, v7.16b, v26.4b[1]",
        ".inst 0x4f9ae8e0  // sdot v0.4s, v7.16b, v26.4b[2]",
        ".inst 0x4fbae8fe  // sdot v30.4s, v7.16b, v26.4b[3]",
        ".inst 0x4f99e064  // sdot v4.4s, v3.16b, v25.4b[0]",
        ".inst 0x4fb9e061  // sdot v1.4s, v3.16b, v25.4b[1]",
        ".inst 0x4f99e860  // sdot v0.4s, v3.16b, v25.4b[2]",
        ".inst 0x4fb9e87e  // sdot v30.4s, v3.16b, v25.4b[3]",
        ".inst 0x4f98e1a4  // sdot v4.4s, v13.16b, v24.4b[0]",
        ".inst 0x4fb8e1a1  // sdot v1.4s, v13.16b, v24.4b[1]",
        ".inst 0x4f98e9a0  // sdot v0.4s, v13.16b, v24.4b[2]",
        ".inst 0x4fb8e9be  // sdot v30.4s, v13.16b, v24.4b[3]",
        ".inst 0x4f90e384  // sdot v4.4s, v28.16b, v16.4b[0]",
        ".inst 0x4fb0e381  // sdot v1.4s, v28.16b, v16.4b[1]",
        ".inst 0x4f90eb80  // sdot v0.4s, v28.16b, v16.4b[2]",
        ".inst 0x4fb0eb9e  // sdot v30.4s, v28.16b, v16.4b[3]",
        "scvtf v4.4s, v4.4s, #0x4",
        "scvtf v1.4s, v1.4s, #0x4",
        "scvtf v0.4s, v0.4s, #0x4",
        "fmla v15.4s, v4.4s, v11.4s",
        "scvtf v30.4s, v30.4s, #0x4",
        "fmla v19.4s, v1.4s, v23.4s",
        "fmla v18.4s, v0.4s, v17.4s",
        "fmla v14.4s, v30.4s, v6.4s",
        "bgt 7b",
        "mov x20, {res_ptr}",
        "cmp x10, #0x1",
        "str q15, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "ble 8f",
        "cmp x10, #0x2",
        "str q19, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "ble 8f",
        "cmp x10, #0x3",
        "str q18, [x20, #0x0]",
        "add x20, x20, {res_stride}",
        "ble 8f",
        "str q14, [x20, #0x0]",
        "8:",  // Row tail: Accumulator store skip
        "subs x23, x23, #0x4",
        "add {res_ptr}, {res_ptr}, #0x10",
        "bne 6b",
        "subs x10, x10, #0x4",
        "add {a_ptr}, {a_ptr}, x9",
        "mov {res_ptr}, x22",
        "bgt 5b",
        "9:",  // Row tail: Row loop skip
        nr = in(reg) nr,
        nb = in(reg) nb,
        b_ptr = inout(reg) b_ptr => _,
        a_ptr = inout(reg) a_ptr => _,
        res_ptr = inout(reg) res_ptr => _,
        res_stride = in(reg) res_stride,
        nc = in(reg) nc,
        out("x9") _, out("x10") _, out("x20") _, out("x21") _, out("x22") _, out("x23") _, out("x24") _, out("x25") _, out("x26") _, out("x27") _, out("x28") _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
        options(nostack, preserves_flags),
    );
}

// Unquantized GEMM function using OpenBLAS
fn gemm_unquantized_openblas(
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
        dgemm(
            b'N',      // transa: 'N' for no transpose
            b'N',      // transb: 'N' for no transpose
            nr as i32, // m: number of rows of matrix A and C
            nc as i32, // n: number of columns of matrix B and C
            n as i32,  // k: number of columns of matrix A and rows of matrix B
            alpha,     // alpha
            vx,        // matrix A
            nr as i32, // lda: leading dimension of A
            vy,        // matrix B
            n as i32,  // ldb: leading dimension of B
            beta,      // beta
            s,         // matrix C
            nr as i32, // ldc: leading dimension of C
        );
    }
}

// Unquantized GEMM function using BLIS
fn gemm_unquantized_blis(n: usize, s: &mut [f32], vx: &[f32], vy: &[f32], nr: usize, nc: usize) {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    unsafe {
        sgemm_(
            &{ b'N' } as *const u8, // transa: 'N' for no transpose
            &{ b'N' } as *const u8, // transb: 'N' for no transpose
            &(nr as i32),           // m: number of rows of matrix A and C
            &(nc as i32),           // n: number of columns of matrix B and C
            &(n as i32),            // k: number of columns of matrix A and rows of matrix B
            &alpha,                 // alpha
            vx.as_ptr(),            // matrix A
            &(nr as i32),           // lda: leading dimension of A
            vy.as_ptr(),            // matrix B
            &(n as i32),            // ldb: leading dimension of B
            &beta,                  // beta
            s.as_mut_ptr(),         // matrix C
            &(nr as i32),           // ldc: leading dimension of C
        );
    }
}

// Unquantized GEMV function
fn gemm_unquantized(n: usize, s: &mut [f32], vx: &[f32], vy: &[f32], nr: usize, nc: usize) {
    for i in 0..nr {
        for j in 0..nc {
            let mut sum = 0.0;
            for k in 0..n {
                sum += vx[i * n + k] * vy[k * nc + j];
            }
            s[i * nc + j] = sum;
        }
    }
}

// Unquantized GEMV function
fn gemm_unquantized_parallel(
    n: usize,
    s: &mut [f32],
    vx: &[f32],
    vy: &[f32],
    nr: usize,
    nc: usize,
) {
    s.par_chunks_mut(nc).enumerate().for_each(|(i, s_row)| {
        for j in 0..nc {
            let mut sum = 0.0;
            for k in 0..n {
                sum += vx[i * n + k] * vy[k * nc + j];
            }
            s_row[j] = sum;
        }
    });
}

// Struct to hold test data
#[derive(Clone, Debug)]
struct TestData {
    vx_404: Vec<BlockQ4_0x4>,
    vy_804: Vec<BlockQ8_0x4>,
    vx_40: Vec<BlockQ4_0>,
    // vy_40: Vec<BlockQ4_0>,
    vx_80: Vec<BlockQ8_0>,
    vy_80: Vec<BlockQ8_0>,
    s: Vec<f32>,
    ux: Vec<f32>,
    uy: Vec<f32>,
}
// Function to generate test data
fn generate_test_data(n: usize, nc: usize, nr: usize, random: bool) -> TestData {
    let nb = n / QK8_0; // Number of blocks along n

    // For A matrix (vy)
    let ux_size = nr * n; // A is nr rows x n cols
    let vy_size = (nr / NCOLS_INTERLEAVED) * nb; // vy will have (nr / 4) * nb blocks

    let ux: Vec<f32> = (0..ux_size)
        .map(|_| {
            if random {
                rand::random::<f32>() - 0.5
            } else {
                1.0f32
            }
        }) // Use actual random data or any desired values
        .collect();

    let mut vy = vec![BlockQ8_0x4::default(); vy_size];

    quantize_q8_0_4x4(&ux, &mut vy, nr, n);

    // For B matrix (vx)
    let uy_size = n * nc; // B is n rows x nc cols
    let vx_size = (nc / NCOLS_INTERLEAVED) * nb; // vx will have (nc / 4) * nb blocks

    let uy: Vec<f32> = (0..uy_size)
        .map(|_| {
            if random {
                rand::random::<f32>() - 0.5
            } else {
                1.0f32
            }
        }) // Use actual random data or any desired values
        .collect();

    let mut vx = vec![BlockQ4_0x4::default(); vx_size];

    quantize_q4_0_4x4(&uy, &mut vx, nc, n); // Note: Ensure that quantize_q4_0_4x4 handles the dimensions correctly

    // **Add dequantization and checks here**

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

    // Dequantize the first block of vy (A matrix)
    let dequantized_vy_block = dequantize_block_q8_0x4(&vy[0]);
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

    // Output s is of size nr x nc
    let s = vec![0.0f32; nr * nc];

    // quantize to 4 bits
    let vx_40 = quantize_f32_to_q4_0(&ux, n, nr);
    // let vy_40 = quantize_f32_to_q4_0(&uy, n, nc);

    let vx_80 = quantize_f32_to_q8_0(&ux, n, nr);
    let vy_80 = quantize_f32_to_q8_0(&uy, n, nc);

    TestData {
        vx_404: vx,
        vy_804: vy,
        vx_40: vx_40,
        // vy_40: vy_40,
        vx_80: vx_80,
        vy_80: vy_80,
        s,
        ux,
        uy,
    }
}

// Benchmark function for scalar implementation
fn benchmark_scalar_implementation(
    iterations: usize,
    n: usize,
    nc: usize,
    nr: usize,
    bs: usize,
    test_data: &TestData,
) {
    let name = "Quantized GEMM scalar";

    // Prepare variables
    let mut s_quantized = vec![0.0f32; nc * nr];
    let mut s_unquantized = vec![0.0f32; nc * nr];

    // Benchmark quantized GEMV (scalar)
    let start_q = Instant::now();
    for _ in 0..iterations {
        gemm_q4_0_4x4_q8_0_scalar_parallel(
            n,
            &mut s_quantized,
            bs,
            &test_data.vx_404,
            &test_data.vy_804,
            nr,
            nc,
        );
    }
    let duration_q = start_q.elapsed();

    // Benchmark unquantized GEMV
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemm_unquantized_parallel(n, &mut s_unquantized, &test_data.ux, &test_data.uy, nr, nc);
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
        "s_unquantized asm: {:?}",
        s_unquantized.to_vec()[s_unquantized.len() - 10..s_unquantized.len()].to_vec()
    );
    println!(
        "s_quantized asm: {:?}",
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

// Benchmark function for scalar implementation
fn benchmark_q4_implementation(
    iterations: usize,
    n: usize,
    nc: usize,
    nr: usize,
    bs: usize,
    test_data: &TestData,
) {
    let name = "Quantized GEMM q4";

    // Prepare variables
    let mut s_quantized = vec![0.0f32; nc * nr];
    let mut s_unquantized = vec![0.0f32; nc * nr];

    // Benchmark quantized GEMV (scalar)
    let start_q = Instant::now();
    for _ in 0..iterations {
        gemm_q4_0_q8_0_scalar(
            n,
            &mut s_quantized,
            bs,
            &test_data.vx_40,
            &test_data.vy_80,
            nr,
            nc,
        );
    }
    let duration_q = start_q.elapsed();

    // Benchmark unquantized GEMV
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemm_unquantized_parallel(n, &mut s_unquantized, &test_data.ux, &test_data.uy, nr, nc);
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
        "s_unquantized q4: {:?}",
        s_unquantized.to_vec()[s_unquantized.len() - 10..s_unquantized.len()].to_vec()
    );
    println!(
        "s_quantized q4: {:?}",
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
        gemm_unquantized_parallel(n, &mut s_unquantized, &test_data.ux, &test_data.uy, nr, nc);
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

// Benchmark function for assembly implementation
fn benchmark_asm_implementation(
    iterations: usize,
    n: usize,
    nc: usize,
    nr: usize,
    bs: usize,
    test_data: &TestData,
) {
    let name = "Quantized GEMM assembly";

    // Prepare variables
    let mut s_quantized = vec![0.0f32; nc * nr];
    let mut s_unquantized = vec![0.0f32; nc * nr];

    // Benchmark quantized GEMV (assembly)
    let start_q = Instant::now();
    for _ in 0..iterations {
        unsafe {
            gemm_q4_0_4x4_q8_0_asm(
                n,
                &mut s_quantized,
                bs,
                &test_data.vx_404,
                &test_data.vy_804,
                nr,
                nc,
            );
        }
    }
    let duration_q = start_q.elapsed();

    // Benchmark unquantized GEMV
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemm_unquantized_parallel(n, &mut s_unquantized, &test_data.ux, &test_data.uy, nr, nc);
    }
    let duration_uq = start_uq.elapsed();

    println!();

    println!(
        "{} (Quantized): {:.2} ms per iteration",
        name,
        (duration_q.as_secs_f64() * 1000.0) / iterations as f64
    );

    println!(
        "Unquantized GEMM (Unquantized): {:.2} ms per iteration",
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
        "s_unquantized asm: {:?}",
        s_unquantized.to_vec()[s_unquantized.len() - 10..s_unquantized.len()].to_vec()
    );
    println!(
        "s_quantized asm: {:?}",
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

    // calculate how many in s_quantized are zero
    let mut zero_count = 0;
    for i in 0..s_quantized.len() {
        if s_quantized[i] == 0.0 {
            zero_count += 1;
        }
    }
    println!("zero count: {}, out of: {}", zero_count, s_quantized.len());
}

// Benchmark function for simd implementation
fn benchmark_simd_implementation(
    iterations: usize,
    n: usize,
    nc: usize,
    nr: usize,
    bs: usize,
    test_data: &TestData,
) {
    let name = "Quantized GEMM Simd";

    // Prepare variables
    let mut s_quantized = vec![0.0f32; nc * nr];
    let mut s_unquantized = vec![0.0f32; nc * nr];

    // Benchmark quantized GEMV (assembly)
    let start_q = Instant::now();
    for _ in 0..iterations {
        gemm_q4_0_4x4_q8_0_simd(
            n,
            &mut s_quantized,
            bs,
            &test_data.vx_404,
            &test_data.vy_804,
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
        "Unquantized GEMM (Unquantized): {:.2} ms per iteration",
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
        "s_unquantized asm: {:?}",
        s_unquantized.to_vec()[s_unquantized.len() - 10..s_unquantized.len()].to_vec()
    );
    println!(
        "s_quantized asm: {:?}",
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

// Benchmark function for OpenBLAS implementation
fn benchmark_openblas_implementation(
    iterations: usize,
    n: usize,
    nc: usize,
    nr: usize,
    test_data: &TestData,
) {
    let name = "Unquantized GEMM OpenBLAS";

    // Prepare variables
    let mut s_unquantized = vec![0.0f32; nc * nr];

    let mut s_unquantized_openblas = vec![0.0f64; nc * nr];

    let ux_openblas: Vec<f64> = test_data.ux.iter().map(|&x| x as f64).collect();
    let uy_openblas: Vec<f64> = test_data.uy.iter().map(|&x| x as f64).collect();

    // Benchmark unquantized GEMM (OpenBLAS)
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemm_unquantized_openblas(
            n,
            &mut s_unquantized_openblas,
            &ux_openblas,
            &uy_openblas,
            nr,
            nc,
        );
    }
    let duration_uq_openblas = start_uq.elapsed();

    // Benchmark unquantized GEMV
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemm_unquantized_parallel(n, &mut s_unquantized, &test_data.ux, &test_data.uy, nr, nc);
    }
    let duration_uq = start_uq.elapsed();

    println!();

    println!(
        "{} (OpenBLAS): {:.2} ms per iteration",
        name,
        (duration_uq_openblas.as_secs_f64() * 1000.0) / iterations as f64
    );

    println!(
        "Unquantized GEMM (Unquantized): {:.2} ms per iteration",
        (duration_uq.as_secs_f64() * 1000.0) / iterations as f64
    );

    println!(
        "s_unquantized openblas: {:?}",
        s_unquantized_openblas[0..4].to_vec()
    );

    println!("Comparing outputs between openblas and unquantized GEMM:");
    let max_diff = s_unquantized_openblas
        .iter()
        .zip(s_unquantized.iter())
        .map(|(&a, &b)| (a - b as f64).abs())
        .fold(0.0, f64::max);
    println!("Maximum difference: {:.6}", max_diff);

    println!(
        "s_unquantized: {:?}",
        s_unquantized.to_vec()[s_unquantized.len() - 10..s_unquantized.len()].to_vec()
    );
    println!(
        "s_unquantized openblas: {:?}",
        s_unquantized_openblas.to_vec()
            [s_unquantized_openblas.len() - 10..s_unquantized_openblas.len()]
            .to_vec()
    );
    println!(
        "diff: {:?}",
        s_unquantized
            .iter()
            .zip(s_unquantized_openblas.iter())
            .map(|(a, b)| *a as f64 - *b)
            .collect::<Vec<f64>>()[(s_unquantized.len() - 10)..s_unquantized.len()]
            .to_vec()
    );
}

// Benchmark function for BLIS implementation
fn benchmark_blis_implementation(
    iterations: usize,
    n: usize,
    nc: usize,
    nr: usize,
    test_data: &TestData,
) {
    let name = "Unquantized GEMM BLIS";

    // Prepare variables
    let mut s_unquantized = vec![0.0f32; nc * nr];

    let mut s_unquantized_blis = vec![0.0f32; nc * nr];

    // Benchmark unquantized GEMM (BLIS)
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemm_unquantized_blis(
            n,
            &mut s_unquantized_blis,
            &test_data.ux,
            &test_data.uy,
            nr,
            nc,
        );
    }
    let duration_uq_blis = start_uq.elapsed();

    // Benchmark unquantized GEMV
    let start_uq = Instant::now();
    for _ in 0..iterations {
        gemm_unquantized_parallel(n, &mut s_unquantized, &test_data.ux, &test_data.uy, nr, nc);
    }
    let duration_uq = start_uq.elapsed();

    println!();

    println!(
        "{} (BLIS): {:.2} ms per iteration",
        name,
        (duration_uq_blis.as_secs_f64() * 1000.0) / iterations as f64
    );

    println!(
        "Unquantized GEMM (Unquantized): {:.2} ms per iteration",
        (duration_uq.as_secs_f64() * 1000.0) / iterations as f64
    );

    println!(
        "s_unquantized blis: {:?}",
        s_unquantized_blis[0..4].to_vec()
    );

    println!("Comparing outputs between blis and unquantized GEMM:");
    let max_diff = s_unquantized_blis
        .iter()
        .zip(s_unquantized.iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .fold(0.0, f64::max);
    println!("Maximum difference: {:.6}", max_diff);

    println!(
        "s_unquantized: {:?}",
        s_unquantized.to_vec()[s_unquantized.len() - 10..s_unquantized.len()].to_vec()
    );
    println!(
        "s_unquantized blis: {:?}",
        s_unquantized_blis.to_vec()[s_unquantized_blis.len() - 10..s_unquantized_blis.len()]
            .to_vec()
    );
    println!(
        "diff: {:?}",
        s_unquantized
            .iter()
            .zip(s_unquantized_blis.iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<f32>>()[(s_unquantized.len() - 10)..s_unquantized.len()]
            .to_vec()
    );
}

fn main() {
    let n = 32; // k
    let nc = 32; // n
    let nr = 4; // m
    let bs = nc;
    let iterations = 1;

    println!(
        "Benchmarking with n={}, nc={}, nr={}, iterations={}",
        n, nc, nr, iterations
    );

    // Generate test data
    let test_data = generate_test_data(n, nc, nr, true);

    // // Benchmark scalar implementation
    // benchmark_scalar_implementation(iterations, n, nc, nr, bs, &test_data);

    // // Benchmark assembly implementation
    // benchmark_asm_implementation(iterations, n, nc, nr, bs, &test_data);

    // // Benchmark simd implementation
    // // benchmark_simd_implementation(iterations, n, nc, nr, bs, &test_data);

    // // Benchmark OpenBLAS implementation
    // benchmark_openblas_implementation(iterations, n, nc, nr, &test_data);

    // // Benchmark BLIS implementation
    // benchmark_blis_implementation(iterations, n, nc, nr, &test_data);

    // benchmark q4
    benchmark_q4_implementation(iterations, n, nc, nr, bs, &test_data);

    // benchmark q8
    benchmark_q8_implementation(iterations, n, nc, nr, bs, &test_data);
}
