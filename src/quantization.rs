use half::f16;
use num::SimdInt;
use rayon::prelude::*;
use std::simd::*;

// Constants
pub const QK8_0: usize = 32;
pub const QK4_0: usize = 32;
pub const BLOCK_SIZE: usize = 4;
pub const NCOLS_INTERLEAVED: usize = 4;

// Data structures

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub d: f16,              // scale (half-precision float)
    pub qs: [u8; QK4_0 / 2], // quantized values (4-bit packed into bytes)
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ4_0x4 {
    pub d: [f16; 4],         // scales (half-precision floats)
    pub qs: [u8; QK4_0 * 2], // quantized values (4-bit packed into bytes)
}

impl Default for BlockQ4_0x4 {
    fn default() -> Self {
        BlockQ4_0x4 {
            d: [f16::default(); 4],
            qs: [0u8; QK4_0 * 2],
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ4_0x8 {
    pub d: [f16; 8],         // scales (half-precision floats)
    pub qs: [u8; QK4_0 * 4], // quantized values (4-bit packed into bytes)
}

impl Default for BlockQ4_0x8 {
    fn default() -> Self {
        BlockQ4_0x8 {
            d: [f16::default(); 8],
            qs: [0u8; QK4_0 * 4],
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub d: f16,          // scale (half-precision float)
    pub qs: [i8; QK8_0], // quantized values (8-bit)
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ8_0x4 {
    pub d: [f16; 4],         // scales (half-precision floats)
    pub qs: [i8; QK8_0 * 4], // quantized values (8-bit)
}

impl Default for BlockQ8_0x4 {
    fn default() -> Self {
        BlockQ8_0x4 {
            d: [f16::default(); 4],
            qs: [0i8; QK8_0 * 4],
        }
    }
}

/*

We need functions to go from:
- f32 to/from BlockQ8_0
- BlockQ8_0 to/from BlockQ8_0x4
- f32 to/from BlockQ4_0
- BlockQ4_0 to/from BlockQ4_0x4
*/

// NEW FUNCTIONS
// For new functions don't use any old functions in the implementation, but you can use the same logic

pub fn quantize_f32_to_q4_0(src: &[f32], nc: usize, nr: usize) -> Vec<BlockQ4_0> {
    assert!(nc % QK4_0 == 0);

    let number_of_blocks = nc / QK4_0;
    let mut dst = vec![BlockQ4_0::default(); number_of_blocks * nr];

    // for each row
    for i in 0..nr {
        // for each block
        for j in 0..number_of_blocks {
            let src_offset = i * nc + j * QK4_0;
            let src_slice = &src[src_offset..src_offset + QK4_0];
            let block = &mut dst[i * number_of_blocks + j];

            let mut max_abs = 0.0f32;
            for &v in src_slice.iter().take(QK4_0) {
                if v.abs() > max_abs {
                    max_abs = v.abs();
                }
            }

            let d = max_abs / 7.0f32; // 4-bit quantization, so 2^3 - 1 = 7
            let id = if d != 0.0f32 { 1.0f32 / d } else { 0.0f32 };

            block.d = f16::from_f32(d);
            for i in (0..QK4_0).step_by(2) {
                let mut q = 0u8;
                for j in 0..2 {
                    let v = src_slice[i + j] * id;
                    let mut vi = v.round() as i32;
                    vi = vi.clamp(-8, 7);
                    let qvi = (vi + 8) as u8;
                    q |= (qvi & 0x0F) << (j * 4);
                }
                block.qs[i / 2] = q;
            }
        }
    }
    dst
}

pub fn unquantize_q4_0_to_f32(src: &[BlockQ4_0], nc: usize, nr: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; nc * nr];

    // for each block
    src.iter().enumerate().for_each(|(block_no, block)| {
        // calculate the index for the dst
        let dst_offset = block_no * QK4_0;
        // for each element in the block
        for i in 0..QK4_0 {
            let q_byte = block.qs[i / 2];
            let q = if i % 2 == 0 {
                (q_byte >> 0) & 0x0F
            } else {
                (q_byte >> 4) & 0x0F
            };
            let q_signed = q as i8 - 8; // Since we added 8 during quantization
            dst[dst_offset + i] = q_signed as f32 * block.d.to_f32();
        }
    });
    dst
}

pub fn quantize_f32_to_q8_0(src: &[f32], nc: usize, nr: usize) -> Vec<BlockQ8_0> {
    assert!(nc % QK8_0 == 0);

    let number_of_blocks = nc / QK8_0;
    let mut dst = vec![BlockQ8_0::default(); number_of_blocks * nr];

    // for each row
    for i in 0..nr {
        // for each block
        for j in 0..number_of_blocks {
            let src_offset = i * nc + j * QK8_0;
            let src_slice = &src[src_offset..src_offset + QK8_0];
            let block = &mut dst[i * number_of_blocks + j];

            let mut max_abs = 0.0f32;
            for &v in src_slice.iter().take(QK8_0) {
                if v.abs() > max_abs {
                    max_abs = v.abs();
                }
            }

            let d = max_abs / 127.0f32;
            let id = if d != 0.0f32 { 1.0f32 / d } else { 0.0f32 };
            block.d = f16::from_f32(d);
            for j in 0..QK8_0 {
                let v = src_slice[j] * id;
                let q = v.round() as i8;
                block.qs[j] = q;
            }
        }
    }
    dst
}

pub fn unquantize_q8_0_to_f32(src: &[BlockQ8_0], nc: usize, nr: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; nc * nr];

    // for each block
    src.iter().enumerate().for_each(|(block_no, block)| {
        // calculate the index for the dst
        let dst_offset = block_no * QK8_0;
        // for each element in the block
        for i in 0..QK8_0 {
            let q = block.qs[i] as i32;
            dst[dst_offset + i] = q as f32 * block.d.to_f32();
        }
    });
    dst
}

pub fn q4_0_to_q4_0_4_4(src: &[BlockQ4_0]) -> Vec<BlockQ4_0x4> {
    let mut dst = vec![BlockQ4_0x4::default(); src.len() / 4];

    // chunk in blocks of 4
    src.chunks(4).enumerate().for_each(|(block_no, block)| {
        let dst_offset = block_no * 4;
        let mut dst_block = vec![BlockQ4_0::default(); 4];
        dst_block.copy_from_slice(block);
        dst[dst_offset] = make_block_q4_0x4(&dst_block, 4, 0x88);
    });

    dst
}

pub fn q4_0_4_4_to_q4_0(src: &[BlockQ4_0x4]) -> Vec<BlockQ4_0> {
    let mut dst = vec![BlockQ4_0::default(); 4 * src.len()];

    //
    src.iter().enumerate().for_each(|(block_no, block)| {
        let dst_offset = block_no * 4;
        let dequantized_block = extract_block_q4_0x4(block, 4, 0x88);
        dst[dst_offset..dst_offset + 4].copy_from_slice(&dequantized_block);
    });

    dst
}

pub fn q8_0_to_q8_0_4_4(src: &[BlockQ8_0]) -> Vec<BlockQ8_0x4> {
    let mut dst = vec![BlockQ8_0x4::default(); src.len() / 4];

    src.chunks(4).enumerate().for_each(|(block_no, block)| {
        let mut dst_block = [BlockQ8_0::default(); 4];
        dst_block.copy_from_slice(block);
        let mut combined_block = BlockQ8_0x4::default();
        for i in 0..4 {
            combined_block.d[i] = dst_block[i].d;
            combined_block.qs[i * QK8_0..(i + 1) * QK8_0].copy_from_slice(&dst_block[i].qs);
        }
        dst[block_no] = combined_block;
    });

    dst
}

pub fn q8_0_4_4_to_q8_0(src: &[BlockQ8_0x4]) -> Vec<BlockQ8_0> {
    let mut dst = vec![BlockQ8_0::default(); 4 * src.len()];

    src.iter().enumerate().for_each(|(block_no, block)| {
        let dst_offset = block_no * 4;
        for i in 0..4 {
            let block_q8_0 = BlockQ8_0 {
                d: block.d[i],
                qs: block.qs[i * QK8_0..(i + 1) * QK8_0].try_into().unwrap(),
            };
            dst[dst_offset + i] = block_q8_0;
        }
    });

    dst
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn get_random_data(nc: usize, nr: usize) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut data = vec![0.0f32; nc * nr];
        for i in 0..nr {
            for j in 0..nc {
                data[i * nc + j] = rng.gen_range(-0.5..0.5);
            }
        }
        data
    }

    #[test]
    fn test_quantize_f32_to_q4_0() {
        let nc = 32;
        let nr = 1;
        let src = get_random_data(nc, nr);
        let quantized = quantize_f32_to_q4_0(&src, nc, nr);
        let dequantized = unquantize_q4_0_to_f32(&quantized, nc, nr);
        for (a, b) in src.iter().zip(dequantized.iter()) {
            assert!((a - b).abs() < 0.1);
        }
    }

    #[test]
    fn test_quantize_f32_to_q8_0() {
        let nc = 32;
        let nr = 1;
        let src = get_random_data(nc, nr);
        let quantized = quantize_f32_to_q8_0(&src, nc, nr);
        let dequantized = unquantize_q8_0_to_f32(&quantized, nc, nr);
        for (a, b) in src.iter().zip(dequantized.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_quantize_q4_0_to_q4_0x4() {
        let nc = 32;
        let nr = 4;
        let src = get_random_data(nc, nr);
        let quantized = quantize_f32_to_q4_0(&src, nc, nr);
        let quantized_4x4 = q4_0_to_q4_0_4_4(&quantized);
        let dequantized = q4_0_4_4_to_q4_0(&quantized_4x4);
        let dequantized = unquantize_q4_0_to_f32(&dequantized, nc, nr);
        for (a, b) in src.iter().zip(dequantized.iter()) {
            assert!((a - b).abs() < 0.1, "{} != {}", a, b);
        }
    }

    #[test]
    fn test_quantize_q8_0_to_q8_0x4() {
        let nc = 32;
        let nr = 4;
        let src = get_random_data(nc, nr);
        let quantized = quantize_f32_to_q8_0(&src, nc, nr);
        let quantized = q8_0_to_q8_0_4_4(&quantized);
        let dequantized = q8_0_4_4_to_q8_0(&quantized);
        let dequantized = unquantize_q8_0_to_f32(&dequantized, nc, nr);
        for (a, b) in src.iter().zip(dequantized.iter()) {
            assert!((a - b).abs() < 0.01, "{} != {}", a, b);
        }
    }
}

// OLD FUNCTIONS

// Function to quantize a single row to q4_0 format
pub fn quantize_row_q4_0_ref(src: &[f32], dst: &mut BlockQ4_0, k: usize) {
    let mut max_abs = 0.0f32;
    for &v in src.iter().take(k) {
        if v.abs() > max_abs {
            max_abs = v.abs();
        }
    }

    let d = max_abs / 7.0f32; // 4-bit quantization, so 2^3 - 1 = 7
    let id = if d != 0.0f32 { 1.0f32 / d } else { 0.0f32 };

    dst.d = f16::from_f32(d);
    for i in (0..k).step_by(2) {
        let mut q = 0u8;
        for j in 0..2 {
            let v = src[i + j] * id;
            let mut vi = v.round() as i32;
            vi = vi.clamp(-8, 7);
            let qvi = (vi + 8) as u8;
            q |= (qvi & 0x0F) << (j * 4);
        }
        dst.qs[i / 2] = q;
    }
}

// Function to make BlockQ40x8 from BlockQ40's
pub fn make_block_q4_0x8(
    input: &[BlockQ4_0], // Expected to be of length at least 8
    block_size_interleave: usize,
    xor_mask: u8,
) -> BlockQ4_0x8 {
    let mut out = BlockQ4_0x8::default();

    for i in 0..8 {
        out.d[i] = input[i].d;
    }

    for i in 0..(QK4_0 * 4) {
        let mut src_offset = (i / (8 * block_size_interleave)) * block_size_interleave;
        let src_id = (i % (8 * block_size_interleave)) / block_size_interleave;
        src_offset += i % block_size_interleave;

        out.qs[i] = input[src_id].qs[src_offset] ^ xor_mask;
    }

    out
}

// Function to make BlockQ40x4 from BlockQ40's
pub fn make_block_q4_0x4(
    input: &[BlockQ4_0], // Expected to be of length at least 4
    block_size_interleave: usize,
    xor_mask: u8,
) -> BlockQ4_0x4 {
    let mut out = BlockQ4_0x4::default();

    for i in 0..4 {
        out.d[i] = input[i].d;
    }

    for i in 0..(QK4_0 * 2) {
        let mut src_offset = (i / (4 * block_size_interleave)) * block_size_interleave;
        let src_id = (i % (4 * block_size_interleave)) / block_size_interleave;
        src_offset += i % block_size_interleave;

        out.qs[i] = input[src_id].qs[src_offset] ^ xor_mask;
    }

    out
}

// Function to quantize a matrix to q4_0_4x4 format
pub fn quantize_q4_0_nr_bl(
    src: &[f32],
    dst: &mut [BlockQ4_0x4],
    nrow: usize,
    n_per_row: usize,
    nrows_interleaved: usize,
    block_size_interleave: usize,
) -> usize {
    assert!(n_per_row % QK4_0 == 0);
    assert!(nrows_interleaved <= 8);

    let nb = n_per_row / QK4_0;
    let mut out_index = 0;

    for b in (0..nrow).step_by(nrows_interleaved) {
        for x in 0..nb {
            let mut dst_tmp = vec![BlockQ4_0::default(); nrows_interleaved];

            for i in 0..nrows_interleaved {
                let src_offset = (b + i) * n_per_row + x * QK4_0;
                let src_slice = &src[src_offset..src_offset + QK4_0];
                quantize_row_q4_0_ref(src_slice, &mut dst_tmp[i], QK4_0);
            }
            if nrows_interleaved == 4 {
                dst[out_index] = make_block_q4_0x4(&dst_tmp, block_size_interleave, 0x88);
            }
            out_index += 1;
        }
    }

    ((nrow * n_per_row) / QK4_0) * std::mem::size_of::<BlockQ4_0>()
}

pub fn quantize_q4_0_4x4(
    src: &[f32],
    dst: &mut [BlockQ4_0x4],
    nrow: usize,
    n_per_row: usize,
) -> usize {
    quantize_q4_0_nr_bl(src, dst, nrow, n_per_row, 4, 4)
}

// Function to quantize q8_0 format for GEMM
pub fn quantize_q8_0_4x4(x: &[f32], y: &mut [BlockQ8_0x4], nrow: usize, n_per_row: usize) {
    assert!(QK8_0 == 32);
    assert!(n_per_row % QK8_0 == 0);
    let nb = n_per_row / QK8_0;

    for row in 0..(nrow / 4) {
        for block in 0..nb {
            let mut input_blocks = [BlockQ8_0::default(); 4];

            for i in 0..4 {
                let offset = (row * 4 + i) * n_per_row + block * QK8_0;
                let src_slice = &x[offset..offset + QK8_0];
                quantize_q8_0_4_block(src_slice, &mut input_blocks[i], QK8_0);
            }

            // Combine the 4 BlockQ80 into a BlockQ80x4
            let mut combined_block = BlockQ8_0x4::default();
            for i in 0..4 {
                combined_block.d[i] = input_blocks[i].d;
                combined_block.qs[i * QK8_0..(i + 1) * QK8_0].copy_from_slice(&input_blocks[i].qs);
            }

            let index = row * nb + block;
            y[index] = combined_block;
        }
    }
}

// Function to quantize q8_0 format
pub fn quantize_q8_0_4_block(x: &[f32], y: &mut BlockQ8_0, k: usize) {
    assert!(QK8_0 == 32);
    assert!(k % QK8_0 == 0);
    let nb = k / QK8_0;

    let mut max_abs = 0.0f32;
    for &v in &x[..QK8_0] {
        if v.abs() > max_abs {
            max_abs = v.abs();
        }
    }

    let d = max_abs / 127.0f32;
    let id = if d != 0.0f32 { 1.0f32 / d } else { 0.0f32 };
    y.d = f16::from_f32(d);
    for j in 0..QK8_0 {
        let v = x[j] * id;
        let q = v.round() as i8;
        y.qs[j] = q;
    }
}

// Function to quantize q8_0 format
pub fn quantize_q8_0_4(x: &[f32], y: &mut [BlockQ8_0], k: usize) {
    assert!(QK8_0 == 32);
    assert!(k % QK8_0 == 0);
    let nb = k / QK8_0;

    for i in 0..nb {
        let offset = i * QK8_0;
        let src_slice = &x[offset..offset + QK8_0];
        quantize_q8_0_4_block(src_slice, &mut y[i], QK8_0);
    }
}

pub fn dequantize_block_q8_0x4(block: &BlockQ8_0x4) -> Vec<f32> {
    let mut data = vec![0.0f32; QK8_0 * 4]; // 4 sub-blocks, each with QK8_0 elements

    for i in 0..4 {
        let d = block.d[i].to_f32();
        for j in 0..QK8_0 {
            let q = block.qs[i * QK8_0 + j] as i32;
            data[i * QK8_0 + j] = q as f32 * d;
        }
    }
    data
}

pub fn dequantize_block_q8_0(block: &BlockQ8_0) -> Vec<f32> {
    let mut data = vec![0.0f32; QK8_0];
    let d = block.d.to_f32();

    for i in 0..QK8_0 {
        let q = block.qs[i] as i32;
        data[i] = q as f32 * d;
    }
    data
}

pub fn extract_block_q4_0x4(
    block: &BlockQ4_0x4,
    block_size_interleave: usize,
    xor_mask: u8,
) -> [BlockQ4_0; 4] {
    let mut output = [BlockQ4_0::default(); 4];
    for i in 0..block_size_interleave {
        output[i].d = block.d[i];
    }
    for i in 0..(QK4_0 * 2) {
        let mut src_offset = (i / (4 * block_size_interleave)) * block_size_interleave;
        let src_id = (i % (4 * block_size_interleave)) / block_size_interleave;
        src_offset += i % block_size_interleave;
        output[src_id].qs[src_offset] = block.qs[i] ^ xor_mask;
    }
    output
}

pub fn dequantize_block_q4_0(block: &BlockQ4_0) -> Vec<f32> {
    let mut data = vec![0.0f32; QK4_0];
    let d = block.d.to_f32();

    for (i, data_elem) in data.iter_mut().enumerate() {
        let q_byte = block.qs[i / 2];
        let q = if i % 2 == 0 {
            (q_byte >> 0) & 0x0F
        } else {
            (q_byte >> 4) & 0x0F
        };
        let q_signed = q as i8 - 8; // Since we added 8 during quantization
        *data_elem = q_signed as f32 * d;
    }
    data
}

pub fn dequantize_block_q4_0x4(block: &BlockQ4_0x4) -> Vec<f32> {
    // Extract original BlockQ40s
    let blocks = extract_block_q4_0x4(block, 4, 0x88);
    let mut data = vec![0.0f32; QK4_0 * 4]; // 4 sub-blocks, each with QK4_0 elements

    for i in 0..4 {
        let dequantized_block = dequantize_block_q4_0(&blocks[i]);
        data[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&dequantized_block);
    }
    data
}
