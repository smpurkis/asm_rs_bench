#include <iostream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cassert>
#include <cstring>
#include <bitset>

#define QK8_0 32
#define QK4_0 32

typedef uint16_t ggml_half;
typedef uint16_t ggml_fp16_t;

#define UNUSED(x) (void)(x)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef __cplusplus
    //  not standard in C++
#    if defined(__GNUC__)
#        define GGML_RESTRICT __restrict__
#    elif defined(__clang__)
#        define GGML_RESTRICT __restrict
#    elif defined(_MSC_VER)
#        define GGML_RESTRICT __restrict
#    else
#        define GGML_RESTRICT
#    endif
#else
#    define GGML_RESTRICT 
#endif


template <int K> constexpr int QK_0() {
    if constexpr (K == 4) {
        return QK4_0;
    }
    if constexpr (K == 8) {
        return QK8_0;
    }
    return -1;
}

typedef __fp16 ggml_fp16_internal_t;

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    ggml_fp16_internal_t tmp;
    memcpy(&tmp, &h, sizeof(ggml_fp16_t));
    return (float)tmp;
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
    ggml_fp16_t res;
    ggml_fp16_internal_t tmp = f;
    memcpy(&res, &tmp, sizeof(ggml_fp16_t));
    return res;
}


template <int K, int N> struct block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

// control size
static_assert(sizeof(block<4, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 2, "wrong block<4,4> size/padding");
static_assert(sizeof(block<4, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<4,8> size/padding");
static_assert(sizeof(block<8, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<8,4> size/padding");
static_assert(sizeof(block<8, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 8, "wrong block<8,8> size/padding");

using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;

#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");


static block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 2 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        const uint64_t xor_mask = 0x8888888888888888ULL;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            uint64_t elems;
            // Using memcpy to avoid unaligned memory accesses
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
        }
    } else if (blck_size_interleave == 4) {
        const uint32_t xor_mask = 0x88888888;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;
                
            uint32_t elems = 0;
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint32_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint32_t));
        }
    }

    return out;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
static block_q4_0x8 make_block_q4_0x8(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x8 out;

    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 4 / blck_size_interleave;
    const uint64_t xor_mask = 0x8888888888888888ULL;

    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
        elems ^= xor_mask;
        memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }

    return out;
}


void quantize_q8_0_4x4(const float * x, void * vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

void quantize_q8_0_4x8(const float * x, void * vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

void quantize_mat_q8_0(const float * x, void * vy, int64_t nrow, int64_t n_per_row, int64_t blck_size_interleave) {
    assert(nrow == 4);
    UNUSED(nrow);
    if (blck_size_interleave == 4) {
        quantize_q8_0_4x4(x, vy, n_per_row);
    } else if (blck_size_interleave == 8) {
        quantize_q8_0_4x8(x, vy, n_per_row);
    } else {
        assert(false);
    }
}


void quantize_row_q4_0_ref(const float *  x, block_q4_0 *  y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}


static size_t quantize_q4_0_nr_bl(const float *  src, void *  dst, int64_t nrow, int64_t n_per_row, int nrows_interleaved, int blck_size_interleave) {
    assert(n_per_row % QK4_0 == 0);
    const int nb = n_per_row / QK4_0;

    void * out_ptr = NULL;
    if (nrows_interleaved == 8) {
        out_ptr = (block_q4_0x8 *) dst;
    }
    else if (nrows_interleaved == 4) {
        out_ptr = (block_q4_0x4 *) dst;
    }
    else if (nrows_interleaved == 1) {
        out_ptr = (block_q4_0 *) dst;
    }
    assert(nrows_interleaved <= 8);
    block_q4_0 dst_tmp[8];

    for (int b = 0; b < (nrow * n_per_row); b += nrows_interleaved * n_per_row) {

        for (int64_t x = 0; x < nb; x++) {

            for (int i  = 0; i < nrows_interleaved; i++ ) {
                quantize_row_q4_0_ref(src + b + i * n_per_row + x * QK4_0, (block_q4_0 *) dst_tmp + i, QK4_0);
            }

            if (nrows_interleaved == 8) {
                *(block_q4_0x8 *) out_ptr = make_block_q4_0x8(dst_tmp, blck_size_interleave);
                out_ptr = (block_q4_0x8 *) out_ptr + 1;
            }
            else if (nrows_interleaved == 4) {
                *(block_q4_0x4 *) out_ptr = make_block_q4_0x4(dst_tmp, blck_size_interleave);
                out_ptr = (block_q4_0x4 *) out_ptr + 1;
            } 
            else if (nrows_interleaved == 1) {
                *(block_q4_0 *) out_ptr = dst_tmp[0];
                out_ptr = (block_q4_0 *) out_ptr + 1;
            }
        }
    }

    return ((nrow * n_per_row) / QK4_0 * sizeof(block_q4_0));
}

size_t quantize_q4_0(const float *  src, void *  dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    UNUSED(quant_weights);
    return quantize_q4_0_nr_bl(src, dst, nrow, n_per_row, 1, 1);
}

size_t quantize_q4_0_4x4(const float *  src, void *  dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    UNUSED(quant_weights);
    return quantize_q4_0_nr_bl(src, dst, nrow, n_per_row, 4, 4);
}

size_t quantize_q4_0_4x8(const float *  src, void *  dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    UNUSED(quant_weights);
    return quantize_q4_0_nr_bl(src, dst, nrow, n_per_row, 4, 8);
}

size_t quantize_q4_0_8x8(const float *  src, void *  dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    UNUSED(quant_weights);
    return quantize_q4_0_nr_bl(src, dst, nrow, n_per_row, 8, 8);
}

static float dequantize_q4_0_scalar(const block_q4_0& block, int offset, bool only_quant) {
    const float d = GGML_FP16_TO_FP32(block.d);
    // the first nibble of the first byte is the start of the first row
    // the second nibble of the first byte is the start of the second row
    bool first_nibble = offset < 16;
    int nibble_offset = offset % 16;
    int16_t quant = block.qs[nibble_offset];
    if (first_nibble) {
        quant = quant & 0xF;
    } else {
        quant = quant >> 4;
    }
    if (only_quant) {
        return quant;
    }
    return d * (quant - 8);
}

void convert_q4_0x4_to_q4_0(const block_q4_0x4* src, block_q4_0* dst, int64_t n) {
    assert(n % QK4_0 == 0);
    const int nb = n / QK4_0;
    const uint32_t xor_mask = 0x88888888;

    for (int i = 0; i < nb; i++) {
        // Handle each group of 4 blocks
        for (int block = 0; block < 4; block++) {
            // Copy delta
            dst[i*4 + block].d = src[i].d[block];
            
            // De-interleave the quantized values
            for (int j = 0; j < QK4_0/2; j++) {
                int src_offset = (j/4)*16 + block*4 + (j%4);
                uint32_t q = src[i].qs[src_offset];
                q ^= (xor_mask >> ((src_offset % 4) * 8)) & 0xFF;
                dst[i*4 + block].qs[j] = q;
            }
        }
    }
}

float dequantize_q8_0x4_scalar(const block_q8_0x4& block, int offset) {
    int block_id = offset / QK8_0;
    int block_offset = offset % QK8_0;
    
    const float d = GGML_FP16_TO_FP32(block.d[block_id]);
    
    // Calculate source offset based on interleave pattern
    int src_offset = (block_offset / 4) * 16 + (block_id * 4) + (block_offset % 4);
    
    return d * block.qs[src_offset];
}

// Save test data to file
void save_gemm_test_data(const char* filename,
                        int n, float* s, size_t bs,
                        const void* vx, const void* vy,
                        int nr, int nc) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }

    // Save input parameters
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&bs, sizeof(size_t), 1, f);
    fwrite(&nr, sizeof(int), 1, f);
    fwrite(&nc, sizeof(int), 1, f);

    // Calculate sizes
    const int qk = QK8_0;
    const int nb = n / qk;
    
    // Save input matrices
    size_t vx_size = sizeof(block_q4_0x4) * nb * (nc / 4);
    size_t vy_size = sizeof(block_q8_0x4) * nb * (nr / 4);
    
    fwrite(vx, 1, vx_size, f);
    fwrite(vy, 1, vy_size, f);

    // Save output matrix s
    size_t s_size = nr * bs * sizeof(float);
    fwrite(s, 1, s_size, f);

    fclose(f);
}

// Load test data from file
bool load_gemm_test_data(const char* filename,
                        int& n, float*& s, size_t& bs,
                        void*& vx, void*& vy,
                        int& nr, int& nc) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file for reading: %s\n", filename);
        return false;
    }

    // Load input parameters
    fread(&n, sizeof(int), 1, f);
    fread(&bs, sizeof(size_t), 1, f);
    fread(&nr, sizeof(int), 1, f);
    fread(&nc, sizeof(int), 1, f);

    // Calculate sizes
    const int qk = QK8_0;
    const int nb = n / qk;
    
    // Allocate and load input matrices
    size_t vx_size = sizeof(block_q4_0x4) * nb * (nc / 4);
    size_t vy_size = sizeof(block_q8_0x4) * nb * (nr / 4);
    
    vx = malloc(vx_size);
    vy = malloc(vy_size);
    
    fread(vx, 1, vx_size, f);
    fread(vy, 1, vy_size, f);

    // Allocate and load output matrix s
    size_t s_size = nr * bs * sizeof(float);
    s = (float*)malloc(s_size);
    fread(s, 1, s_size, f);

    fclose(f);
    return true;
}

// Memory cleanup helper
void free_gemm_test_data(float* s, void* vx, void* vy) {
    free(s);
    free(vx);
    free(vy);
}


static void ggml_gemm_q4_0_4x4_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    // save_gemm_test_data("gemm_test_data.bin", n, s, bs, vx, vy, nr, nc);

    // Calculate sizes
    const int qk = QK8_0;
    const int nb = n / qk;
    size_t vx_size = sizeof(block_q4_0x4) * nb * (nc / 4);
    size_t vy_size = sizeof(block_q8_0x4) * nb * (nr / 4);
    size_t s_size = nr * bs * sizeof(float);
    
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

// #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
//     // if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
//     const void * b_ptr = vx;
//     const void * a_ptr = vy;
//     float * res_ptr = s;
//     size_t res_stride = bs * sizeof(float);

//     __asm__ __volatile__(
//         "mov x10, %x[nr]\n"
//         "mov x9, #0x88\n"
//         "cmp x10, #0x10\n"
//         "mul x9, %x[nb], x9\n"
//         "blt 4f\n"
//         "1:"  // Row loop
//         "add x28, %x[b_ptr], #0x8\n"
//         "mov x27, %x[nc]\n"
//         "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
//         "2:"  // Column loop
//         "add x25, %x[a_ptr], #0x8\n"
//         "movi v15.16b, #0x0\n"
//         "movi v19.16b, #0x0\n"
//         "mov x24, %x[nb]\n"
//         "add x23, x25, x9\n"
//         "movi v18.16b, #0x0\n"
//         "movi v14.16b, #0x0\n"
//         "add x22, x23, x9\n"
//         "movi v11.16b, #0x0\n"
//         "movi v13.16b, #0x0\n"
//         "add x21, x22, x9\n"
//         "movi v23.16b, #0x0\n"
//         "movi v16.16b, #0x0\n"
//         "movi v25.16b, #0x0\n"
//         "movi v7.16b, #0x0\n"
//         "movi v0.16b, #0x0\n"
//         "movi v4.16b, #0x0\n"
//         "movi v5.16b, #0x0\n"
//         "movi v21.16b, #0x0\n"
//         "movi v8.16b, #0x0\n"
//         "movi v1.16b, #0x0\n"
//         "3:"  // Block loop
//         "ldr q3, [x28, #0x0]\n"
//         "ldr q31, [x25, #0x0]\n"
//         "movi v28.16b, #0x4\n"
//         "movi v10.4s, #0x0\n"
//         "ldr q22, [x28, #0x10]\n"
//         "ldr q6, [x25, #0x10]\n"
//         "movi v29.4s, #0x0\n"
//         "movi v9.4s, #0x0\n"
//         "ldr q27, [x28, #0x20]\n"
//         "ldr q30, [x28, #0x30]\n"
//         "movi v20.4s, #0x0\n"
//         "movi v24.16b, #0xf0\n"
//         "ldr d2, [x25, #-0x8]\n"
//         "ldr d26, [x23, #-0x8]\n"
//         "sshl v12.16b, v3.16b, v28.16b\n"
//         "sub x20, x28, #0x8\n"
//         "ldr d17, [x20, #0x0]\n"
//         "and v3.16b, v3.16b, v24.16b\n"
//         "subs x24, x24, #0x1\n"
//         "add x28, x28, #0x48\n"
//         ".inst 0x4f9fe18a  // sdot v10.4s, v12.16b, v31.4b[0]\n"
//         ".inst 0x4fbfe19d  // sdot v29.4s, v12.16b, v31.4b[1]\n"
//         ".inst 0x4f9fe989  // sdot v9.4s, v12.16b, v31.4b[2]\n"
//         ".inst 0x4fbfe994  // sdot v20.4s, v12.16b, v31.4b[3]\n"
//         "sshl v31.16b, v22.16b, v28.16b\n"
//         "and v22.16b, v22.16b, v24.16b\n"
//         "fcvtl v17.4s, v17.4h\n"
//         "fcvtl v2.4s, v2.4h\n"
//         "fcvtl v26.4s, v26.4h\n"
//         ".inst 0x4f86e3ea  // sdot v10.4s, v31.16b, v6.4b[0]\n"
//         ".inst 0x4fa6e3fd  // sdot v29.4s, v31.16b, v6.4b[1]\n"
//         ".inst 0x4f86ebe9  // sdot v9.4s, v31.16b, v6.4b[2]\n"
//         ".inst 0x4fa6ebf4  // sdot v20.4s, v31.16b, v6.4b[3]\n"
//         "sshl v6.16b, v27.16b, v28.16b\n"
//         "sshl v28.16b, v30.16b, v28.16b\n"
//         "and v27.16b, v27.16b, v24.16b\n"
//         "and v30.16b, v30.16b, v24.16b\n"
//         "ldr q24, [x25, #0x20]\n"
//         ".inst 0x4f98e0ca  // sdot v10.4s, v6.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
//         ".inst 0x4f98e8c9  // sdot v9.4s, v6.16b, v24.4b[2]\n"
//         ".inst 0x4fb8e8d4  // sdot v20.4s, v6.16b, v24.4b[3]\n"
//         "ldr q24, [x25, #0x30]\n"
//         ".inst 0x4f98e38a  // sdot v10.4s, v28.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e39d  // sdot v29.4s, v28.16b, v24.4b[1]\n"
//         ".inst 0x4f98eb89  // sdot v9.4s, v28.16b, v24.4b[2]\n"
//         ".inst 0x4fb8eb94  // sdot v20.4s, v28.16b, v24.4b[3]\n"
//         "ldr q24, [x25, #0x40]\n"
//         ".inst 0x4f98e06a  // sdot v10.4s, v3.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
//         ".inst 0x4f98e869  // sdot v9.4s, v3.16b, v24.4b[2]\n"
//         ".inst 0x4fb8e874  // sdot v20.4s, v3.16b, v24.4b[3]\n"
//         "ldr q24, [x25, #0x50]\n"
//         ".inst 0x4f98e2ca  // sdot v10.4s, v22.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e2dd  // sdot v29.4s, v22.16b, v24.4b[1]\n"
//         ".inst 0x4f98eac9  // sdot v9.4s, v22.16b, v24.4b[2]\n"
//         ".inst 0x4fb8ead4  // sdot v20.4s, v22.16b, v24.4b[3]\n"
//         "ldr q24, [x25, #0x60]\n"
//         ".inst 0x4f98e36a  // sdot v10.4s, v27.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
//         ".inst 0x4f98eb69  // sdot v9.4s, v27.16b, v24.4b[2]\n"
//         ".inst 0x4fb8eb74  // sdot v20.4s, v27.16b, v24.4b[3]\n"
//         "ldr q24, [x25, #0x70]\n"
//         "add x25, x25, #0x88\n"
//         ".inst 0x4f98e3ca  // sdot v10.4s, v30.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e3dd  // sdot v29.4s, v30.16b, v24.4b[1]\n"
//         ".inst 0x4f98ebc9  // sdot v9.4s, v30.16b, v24.4b[2]\n"
//         ".inst 0x4fb8ebd4  // sdot v20.4s, v30.16b, v24.4b[3]\n"
//         "fmul v24.4s, v17.4s, v2.s[0]\n"
//         "scvtf v10.4s, v10.4s, #0x4\n"
//         "scvtf v29.4s, v29.4s, #0x4\n"
//         "scvtf v9.4s, v9.4s, #0x4\n"
//         "scvtf v20.4s, v20.4s, #0x4\n"
//         "fmla v15.4s, v10.4s, v24.4s\n"
//         "ldr q24, [x23, #0x0]\n"
//         "fmul v10.4s, v17.4s, v2.s[1]\n"
//         "fmla v19.4s, v29.4s, v10.4s\n"
//         "ldr q10, [x23, #0x10]\n"
//         "fmul v29.4s, v17.4s, v2.s[2]\n"
//         "fmul v2.4s, v17.4s, v2.s[3]\n"
//         "fmla v18.4s, v9.4s, v29.4s\n"
//         "movi v9.4s, #0x0\n"
//         "movi v29.4s, #0x0\n"
//         ".inst 0x4f98e189  // sdot v9.4s, v12.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e19d  // sdot v29.4s, v12.16b, v24.4b[1]\n"
//         "fmla v14.4s, v20.4s, v2.4s\n"
//         "movi v20.4s, #0x0\n"
//         "movi v2.4s, #0x0\n"
//         ".inst 0x4f98e994  // sdot v20.4s, v12.16b, v24.4b[2]\n"
//         ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
//         "ldr q24, [x23, #0x20]\n"
//         ".inst 0x4f8ae3e9  // sdot v9.4s, v31.16b, v10.4b[0]\n"
//         ".inst 0x4faae3fd  // sdot v29.4s, v31.16b, v10.4b[1]\n"
//         ".inst 0x4f8aebf4  // sdot v20.4s, v31.16b, v10.4b[2]\n"
//         ".inst 0x4faaebe2  // sdot v2.4s, v31.16b, v10.4b[3]\n"
//         "ldr q10, [x23, #0x30]\n"
//         ".inst 0x4f98e0c9  // sdot v9.4s, v6.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
//         ".inst 0x4f98e8d4  // sdot v20.4s, v6.16b, v24.4b[2]\n"
//         ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
//         "ldr q24, [x23, #0x40]\n"
//         ".inst 0x4f8ae389  // sdot v9.4s, v28.16b, v10.4b[0]\n"
//         ".inst 0x4faae39d  // sdot v29.4s, v28.16b, v10.4b[1]\n"
//         ".inst 0x4f8aeb94  // sdot v20.4s, v28.16b, v10.4b[2]\n"
//         ".inst 0x4faaeb82  // sdot v2.4s, v28.16b, v10.4b[3]\n"
//         "ldr q10, [x23, #0x50]\n"
//         ".inst 0x4f98e069  // sdot v9.4s, v3.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
//         ".inst 0x4f98e874  // sdot v20.4s, v3.16b, v24.4b[2]\n"
//         ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
//         "ldr q24, [x23, #0x60]\n"
//         ".inst 0x4f8ae2c9  // sdot v9.4s, v22.16b, v10.4b[0]\n"
//         ".inst 0x4faae2dd  // sdot v29.4s, v22.16b, v10.4b[1]\n"
//         ".inst 0x4f8aead4  // sdot v20.4s, v22.16b, v10.4b[2]\n"
//         ".inst 0x4faaeac2  // sdot v2.4s, v22.16b, v10.4b[3]\n"
//         "ldr q10, [x23, #0x70]\n"
//         "add x23, x23, #0x88\n"
//         ".inst 0x4f98e369  // sdot v9.4s, v27.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
//         ".inst 0x4f98eb74  // sdot v20.4s, v27.16b, v24.4b[2]\n"
//         ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
//         "ldr q24, [x22, #0x0]\n"
//         ".inst 0x4f8ae3c9  // sdot v9.4s, v30.16b, v10.4b[0]\n"
//         ".inst 0x4faae3dd  // sdot v29.4s, v30.16b, v10.4b[1]\n"
//         ".inst 0x4f8aebd4  // sdot v20.4s, v30.16b, v10.4b[2]\n"
//         ".inst 0x4faaebc2  // sdot v2.4s, v30.16b, v10.4b[3]\n"
//         "fmul v10.4s, v17.4s, v26.s[0]\n"
//         "scvtf v9.4s, v9.4s, #0x4\n"
//         "scvtf v29.4s, v29.4s, #0x4\n"
//         "scvtf v20.4s, v20.4s, #0x4\n"
//         "scvtf v2.4s, v2.4s, #0x4\n"
//         "fmla v11.4s, v9.4s, v10.4s\n"
//         "ldr q9, [x22, #0x10]\n"
//         "fmul v10.4s, v17.4s, v26.s[1]\n"
//         "fmla v13.4s, v29.4s, v10.4s\n"
//         "ldr d29, [x22, #-0x8]\n"
//         "fmul v10.4s, v17.4s, v26.s[2]\n"
//         "fmul v26.4s, v17.4s, v26.s[3]\n"
//         "fcvtl v29.4s, v29.4h\n"
//         "fmla v23.4s, v20.4s, v10.4s\n"
//         "movi v20.4s, #0x0\n"
//         "movi v10.4s, #0x0\n"
//         "fmla v16.4s, v2.4s, v26.4s\n"
//         "movi v26.4s, #0x0\n"
//         "movi v2.4s, #0x0\n"
//         ".inst 0x4f98e194  // sdot v20.4s, v12.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
//         ".inst 0x4f98e99a  // sdot v26.4s, v12.16b, v24.4b[2]\n"
//         ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
//         "ldr q24, [x22, #0x20]\n"
//         ".inst 0x4f89e3f4  // sdot v20.4s, v31.16b, v9.4b[0]\n"
//         ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
//         ".inst 0x4f89ebfa  // sdot v26.4s, v31.16b, v9.4b[2]\n"
//         ".inst 0x4fa9ebe2  // sdot v2.4s, v31.16b, v9.4b[3]\n"
//         "ldr q9, [x22, #0x30]\n"
//         ".inst 0x4f98e0d4  // sdot v20.4s, v6.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e0ca  // sdot v10.4s, v6.16b, v24.4b[1]\n"
//         ".inst 0x4f98e8da  // sdot v26.4s, v6.16b, v24.4b[2]\n"
//         ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
//         "ldr q24, [x22, #0x40]\n"
//         ".inst 0x4f89e394  // sdot v20.4s, v28.16b, v9.4b[0]\n"
//         ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
//         ".inst 0x4f89eb9a  // sdot v26.4s, v28.16b, v9.4b[2]\n"
//         ".inst 0x4fa9eb82  // sdot v2.4s, v28.16b, v9.4b[3]\n"
//         "ldr q9, [x22, #0x50]\n"
//         ".inst 0x4f98e074  // sdot v20.4s, v3.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e06a  // sdot v10.4s, v3.16b, v24.4b[1]\n"
//         ".inst 0x4f98e87a  // sdot v26.4s, v3.16b, v24.4b[2]\n"
//         ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
//         "ldr q24, [x22, #0x60]\n"
//         ".inst 0x4f89e2d4  // sdot v20.4s, v22.16b, v9.4b[0]\n"
//         ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
//         ".inst 0x4f89eada  // sdot v26.4s, v22.16b, v9.4b[2]\n"
//         ".inst 0x4fa9eac2  // sdot v2.4s, v22.16b, v9.4b[3]\n"
//         "ldr q9, [x22, #0x70]\n"
//         "add x22, x22, #0x88\n"
//         ".inst 0x4f98e374  // sdot v20.4s, v27.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e36a  // sdot v10.4s, v27.16b, v24.4b[1]\n"
//         ".inst 0x4f98eb7a  // sdot v26.4s, v27.16b, v24.4b[2]\n"
//         ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
//         "ldr q24, [x21, #0x0]\n"
//         ".inst 0x4f8ae3d4  // sdot v20.4s, v30.16b, v9.4b[0]\n"
//         ".inst 0x4fa9e3ca  // sdot v10.4s, v30.16b, v9.4b[1]\n"
//         ".inst 0x4f8aebda  // sdot v26.4s, v30.16b, v9.4b[2]\n"
//         ".inst 0x4fa9ebc2  // sdot v2.4s, v30.16b, v9.4b[3]\n"
//         "fmul v9.4s, v17.4s, v29.s[0]\n"
//         "scvtf v20.4s, v20.4s, #0x4\n"
//         "scvtf v10.4s, v10.4s, #0x4\n"
//         "scvtf v26.4s, v26.4s, #0x4\n"
//         "scvtf v2.4s, v2.4s, #0x4\n"
//         "fmla v25.4s, v20.4s, v9.4s\n"
//         "ldr q9, [x21, #0x10]\n"
//         "fmul v20.4s, v17.4s, v29.s[1]\n"
//         "fmla v7.4s, v10.4s, v20.4s\n"
//         "ldr d20, [x21, #-0x8]\n"
//         "fmul v10.4s, v17.4s, v29.s[2]\n"
//         "fmul v29.4s, v17.4s, v29.s[3]\n"
//         "fcvtl v20.4s, v20.4h\n"
//         "fmla v0.4s, v26.4s, v10.4s\n"
//         "movi v26.4s, #0x0\n"
//         "movi v10.4s, #0x0\n"
//         "fmla v4.4s, v2.4s, v29.4s\n"
//         "movi v2.4s, #0x0\n"
//         "movi v29.4s, #0x0\n"
//         ".inst 0x4f98e19a  // sdot v26.4s, v12.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
//         ".inst 0x4f98e982  // sdot v2.4s, v12.16b, v24.4b[2]\n"
//         ".inst 0x4fb8e99d  // sdot v29.4s, v12.16b, v24.4b[3]\n"
//         "ldr q12, [x21, #0x20]\n"
//         "fmul v24.4s, v17.4s, v20.s[0]\n"
//         ".inst 0x4f89e3fa  // sdot v26.4s, v31.16b, v9.4b[0]\n"
//         ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
//         ".inst 0x4f89ebe2  // sdot v2.4s, v31.16b, v9.4b[2]\n"
//         ".inst 0x4fa9ebfd  // sdot v29.4s, v31.16b, v9.4b[3]\n"
//         "ldr q9, [x21, #0x30]\n"
//         "fmul v31.4s, v17.4s, v20.s[1]\n"
//         ".inst 0x4f8ce0da  // sdot v26.4s, v6.16b, v12.4b[0]\n"
//         ".inst 0x4face0ca  // sdot v10.4s, v6.16b, v12.4b[1]\n"
//         ".inst 0x4f8ce8c2  // sdot v2.4s, v6.16b, v12.4b[2]\n"
//         ".inst 0x4face8dd  // sdot v29.4s, v6.16b, v12.4b[3]\n"
//         "ldr q12, [x21, #0x40]\n"
//         "fmul v6.4s, v17.4s, v20.s[2]\n"
//         "fmul v20.4s, v17.4s, v20.s[3]\n"
//         ".inst 0x4f89e39a  // sdot v26.4s, v28.16b, v9.4b[0]\n"
//         ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
//         ".inst 0x4f89eb82  // sdot v2.4s, v28.16b, v9.4b[2]\n"
//         ".inst 0x4fa9eb9d  // sdot v29.4s, v28.16b, v9.4b[3]\n"
//         "ldr q9, [x21, #0x50]\n"
//         ".inst 0x4f8ce07a  // sdot v26.4s, v3.16b, v12.4b[0]\n"
//         ".inst 0x4face06a  // sdot v10.4s, v3.16b, v12.4b[1]\n"
//         ".inst 0x4f8ce862  // sdot v2.4s, v3.16b, v12.4b[2]\n"
//         ".inst 0x4face87d  // sdot v29.4s, v3.16b, v12.4b[3]\n"
//         "ldr q12, [x21, #0x60]\n"
//         ".inst 0x4f89e2da  // sdot v26.4s, v22.16b, v9.4b[0]\n"
//         ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
//         ".inst 0x4f89eac2  // sdot v2.4s, v22.16b, v9.4b[2]\n"
//         ".inst 0x4fa9eadd  // sdot v29.4s, v22.16b, v9.4b[3]\n"
//         "ldr q17, [x21, #0x70]\n"
//         "add x21, x21, #0x88\n"
//         ".inst 0x4f8ce37a  // sdot v26.4s, v27.16b, v12.4b[0]\n"
//         ".inst 0x4face36a  // sdot v10.4s, v27.16b, v12.4b[1]\n"
//         ".inst 0x4f8ceb62  // sdot v2.4s, v27.16b, v12.4b[2]\n"
//         ".inst 0x4faceb7d  // sdot v29.4s, v27.16b, v12.4b[3]\n"
//         ".inst 0x4f91e3da  // sdot v26.4s, v30.16b, v17.4b[0]\n"
//         ".inst 0x4fb1e3ca  // sdot v10.4s, v30.16b, v17.4b[1]\n"
//         ".inst 0x4f91ebc2  // sdot v2.4s, v30.16b, v17.4b[2]\n"
//         ".inst 0x4fb1ebdd  // sdot v29.4s, v30.16b, v17.4b[3]\n"
//         "scvtf v26.4s, v26.4s, #0x4\n"
//         "scvtf v10.4s, v10.4s, #0x4\n"
//         "fmla v5.4s, v26.4s, v24.4s\n"
//         "scvtf v2.4s, v2.4s, #0x4\n"
//         "scvtf v29.4s, v29.4s, #0x4\n"
//         "fmla v21.4s, v10.4s, v31.4s\n"
//         "fmla v8.4s, v2.4s, v6.4s\n"
//         "fmla v1.4s, v29.4s, v20.4s\n"
//         "bgt 3b\n"
//         "mov x20, %x[res_ptr]\n"
//         "subs x27, x27, #0x4\n"
//         "add %x[res_ptr], %x[res_ptr], #0x10\n"
//         "str q15, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q19, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q18, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q14, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q11, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q13, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q23, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q16, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q25, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q7, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q0, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q4, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q5, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q21, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q8, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "str q1, [x20, #0x0]\n"
//         "bne 2b\n"
//         "mov x20, #0x4\n"
//         "sub x10, x10, #0x10\n"
//         "cmp x10, #0x10\n"
//         "mov %x[res_ptr], x26\n"
//         "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
//         "bge 1b\n"
//         "4:"  // Row loop skip
//         "cbz x10, 9f\n"
//         "5:"  // Row tail: Row loop
//         "add x24, %x[b_ptr], #0x8\n"
//         "mov x23, %x[nc]\n"
//         "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
//         "6:"  // Row tail: Column loop
//         "movi v15.16b, #0x0\n"
//         "movi v19.16b, #0x0\n"
//         "add x25, %x[a_ptr], #0x8\n"
//         "mov x21, %x[nb]\n"
//         "movi v18.16b, #0x0\n"
//         "movi v14.16b, #0x0\n"
//         "7:"  // Row tail: Block loop
//         "ldr q7, [x24, #0x0]\n"
//         "ldr q5, [x25, #0x0]\n"
//         "movi v9.16b, #0x4\n"
//         "movi v4.4s, #0x0\n"
//         "ldr q3, [x24, #0x10]\n"
//         "ldr q2, [x25, #0x10]\n"
//         "movi v1.4s, #0x0\n"
//         "movi v0.4s, #0x0\n"
//         "ldr q13, [x24, #0x20]\n"
//         "ldr q31, [x25, #0x20]\n"
//         "movi v30.4s, #0x0\n"
//         "movi v29.16b, #0xf0\n"
//         "ldr q28, [x24, #0x30]\n"
//         "ldr q27, [x25, #0x30]\n"
//         "sshl v20.16b, v7.16b, v9.16b\n"
//         "sub x20, x24, #0x8\n"
//         "ldr q26, [x25, #0x40]\n"
//         "ldr q25, [x25, #0x50]\n"
//         "sshl v17.16b, v3.16b, v9.16b\n"
//         "and v7.16b, v7.16b, v29.16b\n"
//         "ldr q24, [x25, #0x60]\n"
//         "ldr q16, [x25, #0x70]\n"
//         "sshl v22.16b, v13.16b, v9.16b\n"
//         "and v3.16b, v3.16b, v29.16b\n"
//         "ldr d21, [x20, #0x0]\n"
//         "ldr d12, [x25, #-0x8]\n"
//         ".inst 0x4f85e284  // sdot v4.4s, v20.16b, v5.4b[0]\n"
//         ".inst 0x4fa5e281  // sdot v1.4s, v20.16b, v5.4b[1]\n"
//         ".inst 0x4f85ea80  // sdot v0.4s, v20.16b, v5.4b[2]\n"
//         ".inst 0x4fa5ea9e  // sdot v30.4s, v20.16b, v5.4b[3]\n"
//         "sshl v9.16b, v28.16b, v9.16b\n"
//         "subs x21, x21, #0x1\n"
//         "and v13.16b, v13.16b, v29.16b\n"
//         "and v28.16b, v28.16b, v29.16b\n"
//         "add x25, x25, #0x88\n"
//         "add x24, x24, #0x48\n"
//         "fcvtl v21.4s, v21.4h\n"
//         "fcvtl v12.4s, v12.4h\n"
//         ".inst 0x4f82e224  // sdot v4.4s, v17.16b, v2.4b[0]\n"
//         ".inst 0x4fa2e221  // sdot v1.4s, v17.16b, v2.4b[1]\n"
//         ".inst 0x4f82ea20  // sdot v0.4s, v17.16b, v2.4b[2]\n"
//         ".inst 0x4fa2ea3e  // sdot v30.4s, v17.16b, v2.4b[3]\n"
//         "fmul v11.4s, v21.4s, v12.s[0]\n"
//         "fmul v23.4s, v21.4s, v12.s[1]\n"
//         "fmul v17.4s, v21.4s, v12.s[2]\n"
//         ".inst 0x4f9fe2c4  // sdot v4.4s, v22.16b, v31.4b[0]\n"
//         "fmul v6.4s, v21.4s, v12.s[3]\n"
//         ".inst 0x4fbfe2c1  // sdot v1.4s, v22.16b, v31.4b[1]\n"
//         ".inst 0x4f9feac0  // sdot v0.4s, v22.16b, v31.4b[2]\n"
//         ".inst 0x4fbfeade  // sdot v30.4s, v22.16b, v31.4b[3]\n"
//         ".inst 0x4f9be124  // sdot v4.4s, v9.16b, v27.4b[0]\n"
//         ".inst 0x4fbbe121  // sdot v1.4s, v9.16b, v27.4b[1]\n"
//         ".inst 0x4f9be920  // sdot v0.4s, v9.16b, v27.4b[2]\n"
//         ".inst 0x4fbbe93e  // sdot v30.4s, v9.16b, v27.4b[3]\n"
//         ".inst 0x4f9ae0e4  // sdot v4.4s, v7.16b, v26.4b[0]\n"
//         ".inst 0x4fbae0e1  // sdot v1.4s, v7.16b, v26.4b[1]\n"
//         ".inst 0x4f9ae8e0  // sdot v0.4s, v7.16b, v26.4b[2]\n"
//         ".inst 0x4fbae8fe  // sdot v30.4s, v7.16b, v26.4b[3]\n"
//         ".inst 0x4f99e064  // sdot v4.4s, v3.16b, v25.4b[0]\n"
//         ".inst 0x4fb9e061  // sdot v1.4s, v3.16b, v25.4b[1]\n"
//         ".inst 0x4f99e860  // sdot v0.4s, v3.16b, v25.4b[2]\n"
//         ".inst 0x4fb9e87e  // sdot v30.4s, v3.16b, v25.4b[3]\n"
//         ".inst 0x4f98e1a4  // sdot v4.4s, v13.16b, v24.4b[0]\n"
//         ".inst 0x4fb8e1a1  // sdot v1.4s, v13.16b, v24.4b[1]\n"
//         ".inst 0x4f98e9a0  // sdot v0.4s, v13.16b, v24.4b[2]\n"
//         ".inst 0x4fb8e9be  // sdot v30.4s, v13.16b, v24.4b[3]\n"
//         ".inst 0x4f90e384  // sdot v4.4s, v28.16b, v16.4b[0]\n"
//         ".inst 0x4fb0e381  // sdot v1.4s, v28.16b, v16.4b[1]\n"
//         ".inst 0x4f90eb80  // sdot v0.4s, v28.16b, v16.4b[2]\n"
//         ".inst 0x4fb0eb9e  // sdot v30.4s, v28.16b, v16.4b[3]\n"
//         "scvtf v4.4s, v4.4s, #0x4\n"
//         "scvtf v1.4s, v1.4s, #0x4\n"
//         "scvtf v0.4s, v0.4s, #0x4\n"
//         "fmla v15.4s, v4.4s, v11.4s\n"
//         "scvtf v30.4s, v30.4s, #0x4\n"
//         "fmla v19.4s, v1.4s, v23.4s\n"
//         "fmla v18.4s, v0.4s, v17.4s\n"
//         "fmla v14.4s, v30.4s, v6.4s\n"
//         "bgt 7b\n"
//         "mov x20, %x[res_ptr]\n"
//         "cmp x10, #0x1\n"
//         "str q15, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "ble 8f\n"
//         "cmp x10, #0x2\n"
//         "str q19, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "ble 8f\n"
//         "cmp x10, #0x3\n"
//         "str q18, [x20, #0x0]\n"
//         "add x20, x20, %x[res_stride]\n"
//         "ble 8f\n"
//         "str q14, [x20, #0x0]\n"
//         "8:"  // Row tail: Accumulator store skip
//         "subs x23, x23, #0x4\n"
//         "add %x[res_ptr], %x[res_ptr], #0x10\n"
//         "bne 6b\n"
//         "subs x10, x10, #0x4\n"
//         "add %x[a_ptr], %x[a_ptr], x9\n"
//         "mov %x[res_ptr], x22\n"
//         "bgt 5b\n"
//         "9:"  // Row tail: Row loop skip
//         : [a_ptr] "+&r" (a_ptr), [res_ptr] "+&r" (res_ptr)
//         : [b_ptr] "r" (b_ptr), [nr] "r" (nr), [nb] "r" (nb), [res_stride] "r" (res_stride), [nc] "r" (nc)
//         : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
//     );
//     // save_gemm_test_data("gemm_test_data_output.bin", n, s, bs, vx, vy, nr, nc);
//     return;
//     // }
// #endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
    {
        float sumf[4][4];
        int sumi;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                for (int l = 0; l < nb; l++) {
                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                for (int i = 0; i < blocklen; ++i) {
                                    const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                    const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                                }
                                sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
    // save_gemm_test_data("gemm_test_data_output.bin", n, s, bs, vx, vy, nr, nc);

    // Print the first 3 values of s
    std::cout << "First 3 values of s: ";
    for (int i = 0; i < 3 && i < nr * bs; ++i) {
        std::cout << s[i] << " ";
    }
    std::cout << "\n";
    std::cout << "\n";
}

void unquantized_matmul(int n, float* s, size_t bs, const float* x, const float* y, int nr, int nc) {
    // Clear the result matrix s
    for (int i = 0; i < nr * bs; ++i) {
        s[i] = 0.0f;
    }

    // Perform matrix multiplication
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nc; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                // x is of shape (nr x n), y is of shape (n x nc)
                std::cout << "i: " << i << " j: " << j << " k: " << k << " x[i * n + k]: " << x[i * n + k] << " y[k * nc + j]: " << y[k * nc + j] << "\n";
                sum += x[i * n + k] * y[k * nc + j];
            }
            // s is of shape (nr x nc) 
            s[i * bs + j] = sum;
        }
    }

    // Print the first 3 values of the result matrix s
    std::cout << "First 3 values of the result matrix unquantized s: ";
    for (int i = 0; i < 3 && i < nr * bs; ++i) {
        std::cout << s[i] << " ";
    }
    std::cout << "\n";
}

// Function to create unquantized test data
void create_unquantized_test_data(int n, int nr, int nc, float * x, float * y, block_q4_0x4* vx, block_q8_0x4* vy) {
    // Set random seed for reproducibility
    srand(42);

    // Initialize the matrices with some test data between -1 and 1
    for (int i = 0; i < n * nc; ++i) {
        x[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
        // x[i] = 1.0f;
    }
    for (int i = 0; i < n * nr; ++i) {
        y[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
        // y[i] = 1.0f;
    }
    std::cout << "x: ";
    for (int i = 0; i < n * nc; ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << "\n";

    std::cout << "y: ";
    for (int i = 0; i < n * nr; ++i) {
        std::cout << y[i] << " ";
    }
    std::cout << "\n";
    std::cout << "\n";

    // Quantize the matrices
    // vy to be quantized to block_q8_0x4
    // vx to be quantized to block_q4_0x4
    quantize_mat_q8_0(y, vy, nr, n, 4);
    quantize_q4_0_4x4(x, vx, nc, n, nullptr); 
    
    std::vector<block_q4_0> vx_tmp(n * nc / QK4_0);

    // Convert q4_0x4 to q4_0
    convert_q4_0x4_to_q4_0(vx, vx_tmp.data(), n * nc);

    // Print the first few elements of the converted q4_0 data for verification
    std::cout << "Converted q4_0 data: ";
    for (int i = 0; i < n * nc; ++i) {
        if (i % QK4_0 == 0) {
            std::cout << "\n";
        }
        float dequantized_value = dequantize_q4_0_scalar(vx_tmp[i / QK4_0], i % QK4_0, false);
        std::cout << dequantized_value << " ";
    }
    std::cout << "\n";

    // Print the first few elements of the q8_0 data for verification
    std::cout << "q8_0 data: ";
    for (int i = 0; i < n * nr; ++i) {
        if (i % QK8_0 == 0) {
            std::cout << "\n";
        }
        float dequantized_value = dequantize_q8_0x4_scalar(vy[i / QK8_0], i % QK8_0);
        std::cout << dequantized_value << " ";
    }
}

// Test harness to run the isolated GEMM function
int main() {

    // const char* test_data_file = "gemm_test_data.bin";
    
    // Variables to hold the test data
    int n;
    // float* s = nullptr;
    std::vector<float> s(4 * 32);
    size_t bs;
    int nr, nc;

    // Load the test data
    // if (!load_gemm_test_data(test_data_file, n, s, bs, vx, vy, nr, nc)) {
    //     std::cerr << "Failed to load test data\n";
    //     return 1;
    // }
    nr = 4;
    nc = 4;
    n = 32;
    bs = nc;

    std::vector<block_q4_0x4> vx(n * nc / QK4_0);
    std::vector<block_q8_0x4> vy(n * nr / QK8_0);
    
    std::cout << "nr: " << nr << " nc: " << nc << " n: " << n << " bs: " << bs << "\n";

    // Create a copy of the original output for comparison
    size_t s_size = nr * nc * sizeof(float);
    
    // Allocate memory for the unquantized output
    // float* s_unquantized = (float*)malloc(s_size);
    std::vector<float> s_unquantized(s_size);

    // Generate unquantized test data for vx and vy
    std::vector<float> x(n * nc);
    std::vector<float> y(n * nr);
    create_unquantized_test_data(n, nr, nc, x.data(), y.data(), vx.data(), vy.data());

    std::cout << "get here" << "\n";

    // Run the unquantized GEMM function
    unquantized_matmul(n, s_unquantized.data(), bs, x.data(), y.data(), nr, nc);
    std::cout << "get here" << "\n";

    // Run the quantized GEMM function
    ggml_gemm_q4_0_4x4_q8_0(n, s.data(), bs, vx.data(), vy.data(), nr, nc);
    std::cout << "get here" << "\n";

    // Compare results
    bool matches = true;
    float max_diff = 0.0f;
    for (int i = 0; i < nr * bs; i++) {
        float diff = std::abs(s[i] - s_unquantized[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-2) {
            matches = false;
            std::cout << "Mismatch at index " << i << ": "
                     << s[i] << " vs " << s_unquantized[i]
                     << " (diff: " << diff << ")\n";
        }
    }

    if (matches) {
        std::cout << "Test passed! Maximum difference: " << max_diff << "\n";
    } else {
        std::cout << "Test failed! Maximum difference: " << max_diff << "\n";
    }
    

    // Cleanup
    // Print the s_unquantized matrix
    std::cout << "s_unquantized matrix: ";
    for (int i = 0; i < nr * bs; ++i) {
        if (i % bs == 0) {
            std::cout << "\n";
        }
        std::cout << s_unquantized[i] << " ";
    }
    std::cout << "\n";

    return matches ? 0 : 1;
}
