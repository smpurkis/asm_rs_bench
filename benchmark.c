// Include necessary headers  
#include <stdio.h>  
#include <stdint.h>  
#include <stdlib.h>  
#include <string.h>   // For memset  
#include <time.h>  
#include <math.h>  
#include <assert.h>  
  
// Constants  
#define QK8_0 32  
#define QK4_0 32  
#define BLOCK_SIZE 4  
#define NCOLS_INTERLEAVED 4  
  
// Define ggml_fp16_t as uint16_t (half-precision floating-point)  
typedef uint16_t ggml_fp16_t;  
  
// Function to convert ggml_fp16_t to float  
static inline float GGML_FP16_TO_FP32(ggml_fp16_t h) {  
    uint16_t h_exp = (h & 0x7C00) >> 10;  
    uint16_t h_sig = h & 0x03FF;  
    uint32_t f_sgn = (h & 0x8000) << 16;  
    uint32_t f_exp, f_sig;  
  
    if (h_exp == 0) {  
        // Zero / subnormal  
        if (h_sig == 0) {  
            // Zero  
            f_exp = 0;  
            f_sig = 0;  
        } else {  
            // Subnormal  
            h_exp = 1;  
            while ((h_sig & 0x0400) == 0) {  
                h_sig <<= 1;  
                h_exp -= 1;  
            }  
            h_sig &= 0x03FF;  
            f_exp = (h_exp + (127 - 15)) << 23;  
            f_sig = h_sig << 13;  
        }  
    } else if (h_exp == 0x1F) {  
        // Inf / NaN  
        f_exp = 0xFF << 23;  
        f_sig = h_sig << 13;  
    } else {  
        // Normalized  
        f_exp = (h_exp + (127 - 15)) << 23;  
        f_sig = h_sig << 13;  
    }  
  
    uint32_t f = f_sgn | f_exp | f_sig;  
    float result = *((float *)&f);  
    return result;  
}  
  
// Function to convert float to ggml_fp16_t  
static inline ggml_fp16_t GGML_FP32_TO_FP16(float f) {  
    uint32_t x = *((uint32_t *)&f);  
    uint16_t h = 0;  
  
    uint32_t sign = (x >> 31) & 0x1;  
    uint32_t mantissa = x & 0x7FFFFF;  
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127;  
  
    if (exp > 15) {  
        // Overflow, set to Inf  
        h = (sign << 15) | (0x1F << 10);  
    } else if (exp > -15) {  
        // Normalized number  
        exp += 15;  
        mantissa >>= 13;  
        h = (sign << 15) | (exp << 10) | (mantissa & 0x3FF);  
    } else {  
        // Underflow, set to zero  
        h = 0;  
    }  
  
    return h;  
}  
  
// Data structures  
  
typedef struct {  
    ggml_fp16_t d;        // scale (half-precision float)  
    uint8_t qs[QK4_0 / 2];  // quantized values (4-bit packed into bytes)  
} block_q4_0;  
  
typedef struct {  
    ggml_fp16_t d[4];        // scales (half-precision floats)  
    uint8_t qs[QK4_0 * 2];   // quantized values (4-bit packed into bytes)  
} block_q4_0x4;  
  
typedef struct {  
    ggml_fp16_t d[8];        // scales (half-precision floats)  
    uint8_t qs[QK4_0 * 4];   // quantized values (4-bit packed into bytes)  
} block_q4_0x8;  
  
typedef struct {  
    ggml_fp16_t d;        // scale (half-precision float)  
    int8_t qs[QK8_0];     // quantized values (8-bit)  
} block_q8_0;  
  
// Utility macros  
#define UNUSED(x) (void)(x)  
#define MAX(a, b) (((a) > (b)) ? (a) : (b))  
  
// Function prototypes  
void gemv_q4_0_4x4_q8_0_scalar(  
    int n,  
    float * restrict s,  
    size_t bs,  
    const block_q4_0x4 * restrict vx,  
    const block_q8_0 * restrict vy,  
    int nr,  
    int nc  
);  
  
void gemv_unquantized(  
    int n,  
    float * restrict s,  
    const float * restrict vx,  
    const float * restrict vy,  
    int nr,  
    int nc  
);  
  
// Functions for quantization  
  
// Function to quantize a single row to q4_0 format  
void quantize_row_q4_0_ref(const float * restrict src, block_q4_0 * restrict dst, int k) {  
    float max_abs = 0.0f;  
    for (int i = 0; i < k; ++i) {  
        float v = src[i];  
        if (fabsf(v) > max_abs) {  
            max_abs = fabsf(v);  
        }  
    }  
  
    const float d = max_abs / 7.0f;  // 4-bit quantization, so 2^3 - 1 = 7  
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;  
  
    dst->d = GGML_FP32_TO_FP16(d);  
    for (int i = 0; i < k; i += 2) {  
        uint8_t q = 0;  
        for (int j = 0; j < 2; ++j) {  
            float v = src[i + j] * id;  
            int vi = (int)(roundf(v));  
            vi = (vi < -8) ? -8 : (vi > 7) ? 7 : vi;  
            uint8_t qvi = (uint8_t)(vi + 8);  
            q |= (qvi & 0x0F) << (j * 4);  
        }  
        dst->qs[i / 2] = q;  
    }  
}  
  
// Function to make block_q4_0x4 from block_q4_0's  
static block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, unsigned int blck_size_interleave, unsigned int xor_mask) {  
    block_q4_0x4 out;  
  
    for (int i = 0; i < 4; i++) {  
        out.d[i] = in[i].d;  
    }  
  
    for (int i = 0; i < QK4_0 * 2; i++) {  
        int src_offset = (i / (4 * blck_size_interleave)) * blck_size_interleave;  
        int src_id = (i % (4 * blck_size_interleave)) / blck_size_interleave;  
        src_offset += (i % blck_size_interleave);  
  
        out.qs[i] = in[src_id].qs[src_offset] ^ xor_mask;  
    }  
  
    return out;  
}  
  
// Function to quantize a matrix to q4_0_4x4 format  
static size_t quantize_q4_0_nr_bl(const float * restrict src, void * restrict dst, int64_t nrow, int64_t n_per_row, int nrows_interleaved, int blck_size_interleave) {  
    assert(n_per_row % QK4_0 == 0);  
    const int nb = n_per_row / QK4_0;  
      
    block_q4_0x4 * out_ptr = (block_q4_0x4 *) dst;  
    assert(nrows_interleaved <= 4);  
  
    block_q4_0 dst_tmp[4];  
  
    for (int b = 0; b < nrow; b += nrows_interleaved) {  
  
        for (int64_t x = 0; x < nb; x++) {  
  
            for (int i  = 0; i < nrows_interleaved; i++ ) {  
                quantize_row_q4_0_ref(src + (b + i) * n_per_row + x * QK4_0, (block_q4_0 *) &dst_tmp[i], QK4_0);  
            }  
  
            *(block_q4_0x4 *) out_ptr = make_block_q4_0x4(dst_tmp, blck_size_interleave, 0x88);  
            out_ptr += 1;  
        }  
    }  
  
    return ((nrow * n_per_row) / QK4_0 * sizeof(block_q4_0));  
}  
  
size_t quantize_q4_0_4x4(const float * restrict src, void * restrict dst, int64_t nrow, int64_t n_per_row) {  
    return quantize_q4_0_nr_bl(src, dst, nrow, n_per_row, 4, 4);  
}  
  
// Function to quantize q8_0 format  
void quantize_q8_0_4x4(const float * restrict x, void * restrict vy, int64_t k) {  
    assert(QK8_0 == 32);  
    assert(k % QK8_0 == 0);  
    const int nb = k / QK8_0;  
  
    block_q8_0 * restrict y = (block_q8_0 *) vy;  
  
    for (int i = 0; i < nb; i++) {  
        float max_abs = 0.0f;  
        for (int j = 0; j < QK8_0; j++) {  
            float v = x[i * QK8_0 + j];  
            if (fabsf(v) > max_abs) {  
                max_abs = fabsf(v);  
            }  
        }  
        float d = max_abs / 127.0f;  
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;  
        y[i].d = GGML_FP32_TO_FP16(d);  
        for (int j = 0; j < QK8_0; j++) {  
            float v = x[i * QK8_0 + j] * id;  
            int8_t q = (int8_t)(roundf(v));  
            y[i].qs[j] = q;  
        }  
    }  
}  

// Assembly function provided by the user  
void ggml_gemv_q4_0_4x4_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, const void * restrict vy, int nr, int nc) {  
    const int qk = QK8_0;  
    const int nb = n / qk;  
    const int ncols_interleaved = 4;  
    const int blocklen = 4;  
  
    assert (n % qk == 0);  
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
    // Print metadata inputs and variables  
    // printf("Metadata Inputs and Variables:\n");  
    // printf("n: %d\n", n);  
    // printf("bs: %zu\n", bs);  
    // printf("nr: %d\n", nr);  
    // printf("nc: %d\n", nc);  
  
  
#if defined(__aarch64__) && defined(__ARM_NEON)  
    {  
        const void * b_ptr = vx;  
        const void * a_ptr = vy;  
        float * res_ptr = s;  
  
        __asm__ __volatile__(  
            "movi v31.16b, #0x4\n"  
            "movi v30.16b, #0xf0\n"  
            "add %x[b_ptr], %x[b_ptr], #0x8\n"  
            "1:"  // Column loop  
            "add x22, %x[a_ptr], #0x2\n"  
            "movi v29.16b, #0x0\n"  
            "mov x21, %x[nb]\n"  
            "2:"  // Block loop  
            "ldr q28, [%x[b_ptr], #0x0]\n"  
            "ldr q27, [x22, #0x0]\n"  
            "movi v26.4s, #0x0\n"  
            "sub x20, x22, #0x2\n"  
            "ldr q25, [x22, #0x10]\n"  
            "ldr q24, [%x[b_ptr], #0x10]\n"  
            "sub x21, x21, #0x1\n"  
            "add x22, x22, #0x22\n"  
            "ldr q23, [%x[b_ptr], #0x20]\n"  
            "ldr q22, [%x[b_ptr], #0x30]\n"  
            "ld1r { v21.8h }, [x20]\n"  
            "ldr q20, [%x[b_ptr], #-0x8]\n"  
            "sshl v16.16b, v28.16b, v31.16b\n"  
            "and v28.16b, v28.16b, v30.16b\n"  
            "sshl v19.16b, v24.16b, v31.16b\n"  
            "and v24.16b, v24.16b, v30.16b\n"  
            "add %x[b_ptr], %x[b_ptr], #0x48\n"  
            "sshl v18.16b, v23.16b, v31.16b\n"  
            "and v23.16b, v23.16b, v30.16b\n"  
            ".inst 0x4f9be21a  // sdot v26.4s, v16.16b, v27.4b[0]\n"  
            "sshl v17.16b, v22.16b, v31.16b\n"  
            "and v22.16b, v22.16b, v30.16b\n"  
            "fcvtl v21.4s, v21.4h\n"  
            "fcvtl v16.4s, v20.4h\n"  
            ".inst 0x4f99e39a  // sdot v26.4s, v28.16b, v25.4b[0]\n"  
            "fmul v16.4s, v16.4s, v21.4s\n"  
            ".inst 0x4fbbe27a  // sdot v26.4s, v19.16b, v27.4b[1]\n"  
            ".inst 0x4fb9e31a  // sdot v26.4s, v24.16b, v25.4b[1]\n"  
            ".inst 0x4f9bea5a  // sdot v26.4s, v18.16b, v27.4b[2]\n"  
            ".inst 0x4f99eafa  // sdot v26.4s, v23.16b, v25.4b[2]\n"  
            ".inst 0x4fbbea3a  // sdot v26.4s, v17.16b, v27.4b[3]\n"  
            ".inst 0x4fb9eada  // sdot v26.4s, v22.16b, v25.4b[3]\n"  
            "scvtf v26.4s, v26.4s, #0x4\n"  
            "fmla v29.4s, v26.4s, v16.4s\n"  
            "cbnz x21, 2b\n"  
            "sub %x[nc], %x[nc], #0x4\n"  
            "str q29, [%x[res_ptr], #0x0]\n"  
            "add %x[res_ptr], %x[res_ptr], #0x10\n"  
            "cbnz %x[nc], 1b\n"  
            : [b_ptr] "+&r" (b_ptr), [res_ptr] "+&r" (res_ptr), [nc] "+&r" (nc)  
            : [a_ptr] "r" (a_ptr), [nb] "r" (nb)  
            : "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22"  
            );  
        return;  
    }  
#endif // #if defined(__aarch64__) && defined(__ARM_NEON)  
  
    // Fallback scalar implementation  
    float sumf[4];  
    int sumi;  
  
    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;  
    for (int x = 0; x < nc / ncols_interleaved; x++) {  
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);  
  
        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;  
        for (int l = 0; l < nb; l++) {  
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {  
                for (int j = 0; j < ncols_interleaved; j++) {  
                    sumi = 0;  
                    for (int i = 0; i < blocklen; ++i) {  
                        int idx = k * ncols_interleaved * blocklen + j * blocklen + i;  
  
                        uint8_t byte = b_ptr[l].qs[idx];  
  
                        // Extract low and high nibbles  
                        int8_t v0 = (byte & 0x0F);  
                        int8_t v1 = (byte & 0xF0) >> 4;  
  
                        // Sign-extend from 4-bit to 8-bit  
                        v0 = (v0 ^ 0x8) - 0x8;  
                        v1 = (v1 ^ 0x8) - 0x8;  
  
                        int a_idx0 = k * blocklen + i;  
                        int a_idx1 = k * blocklen + i + qk / 2;  
  
                        int8_t a0 = a_ptr[l].qs[a_idx0];  
                        int8_t a1 = a_ptr[l].qs[a_idx1];  
  
                        // Multiply and accumulate  
                        sumi += v0 * a0 + v1 * a1;  
                    }  
                    float b_scale = GGML_FP16_TO_FP32(b_ptr[l].d[j]);  
                    float a_scale = GGML_FP16_TO_FP32(a_ptr[l].d);  
  
                    // Adjust sumi by shifting right by 4  
                    sumf[j] += ((float) (sumi >> 4)) * b_scale * a_scale;  
                }  
            }  
        }  
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];  
    }  
}  
  
// Scalar GEMV function for quantized data  
void gemv_q4_0_4x4_q8_0_scalar(  
    int n,  
    float * restrict s,  
    size_t bs,  
    const block_q4_0x4 * restrict vx,  
    const block_q8_0 * restrict vy,  
    int nr,  
    int nc  
) {  
    const int qk = QK8_0;  
    const int nb = n / qk;  
    float sumf[NCOLS_INTERLEAVED];  
  
    assert(n % qk == 0);  
    assert(nc % NCOLS_INTERLEAVED == 0);  
  
    for (int x = 0; x < nc / NCOLS_INTERLEAVED; x++) {  
        // Clear accumulators  
        for (int j = 0; j < NCOLS_INTERLEAVED; j++) {  
            sumf[j] = 0.0f;  
        }  
  
        for (int l = 0; l < nb; l++) {  
            const block_q4_0x4 *b_block = &vx[x * nb + l];  
            const block_q8_0 *a_block = &vy[l];  
  
            for (int k = 0; k < (qk / (2 * BLOCK_SIZE)); k++) {  
                for (int j = 0; j < NCOLS_INTERLEAVED; j++) {  
                    int sumi = 0;  
  
                    for (int i = 0; i < BLOCK_SIZE; i++) {  
                        int idx = k * NCOLS_INTERLEAVED * BLOCK_SIZE + j * BLOCK_SIZE + i;  
  
                        uint8_t byte = b_block->qs[idx];  
  
                        // Extract low and high nibbles and sign extend  
                        int8_t v0 = (int8_t)(byte << 4);       // Get lower nibble and shift left  
                        int8_t v1 = (int8_t)(byte & 0xF0);     // Get upper nibble  
  
                        int a_idx0 = k * BLOCK_SIZE + i;  
                        int a_idx1 = k * BLOCK_SIZE + i + qk / 2;  
  
                        int8_t a0 = a_block->qs[a_idx0];  
                        int8_t a1 = a_block->qs[a_idx1];  
  
                        // Multiply and accumulate with right shift applied to each product  
                        sumi += ((v0 * a0) >> 4) + ((v1 * a1) >> 4);  
                    }  
  
                    float b_scale = GGML_FP16_TO_FP32(b_block->d[j]);  
                    float a_scale = GGML_FP16_TO_FP32(a_block->d);  
  
                    sumf[j] += ((float) sumi) * b_scale * a_scale;  
                }  
            }  
        }  
  
        // Store results  
        for (int j = 0; j < NCOLS_INTERLEAVED; j++) {  
            s[x * NCOLS_INTERLEAVED + j] = sumf[j];  
        }  
    }  
}  
  
// Unquantized GEMV function  
void gemv_unquantized(  
    int n,  
    float * restrict s,  
    const float * restrict vx,  
    const float * restrict vy,  
    int nr,  
    int nc  
) {  
    // Assume 'vx' is a matrix of size [nc x n]  
    // 'vy' is a vector of size [n]  
    // 's' is the result vector of size [nc]  
  
    for (int i = 0; i < nc; ++i) {  
        float sum = 0.0f;  
        for (int k = 0; k < n; ++k) {  
            sum += vx[i * n + k] * vy[k];  
        }  
        s[i] = sum;  
    }  
}  
  
// Function to generate test data  
void generate_test_data(int n, int nc, block_q4_0x4 **vx_out, block_q8_0 **vy_out, float **s_out, float **unquantized_vx, float **unquantized_vy) {  
    int nb = n / QK8_0;  
    int num_vx_blocks = nb * (nc / NCOLS_INTERLEAVED);  
    block_q4_0x4 *vx = (block_q4_0x4 *) malloc(num_vx_blocks * sizeof(block_q4_0x4));  
    block_q8_0 *vy = (block_q8_0 *) malloc(nb * sizeof(block_q8_0));  
    float *s = (float *) malloc(nc * sizeof(float));  
  
    // Generate unquantized data  
    float *ux = (float *) malloc(nc * n * sizeof(float));  
    float *uy = (float *) malloc(n * sizeof(float));  
  
    // Initialize random seed  
    srand(42);  
  
    // Generate random data for ux and uy  
    for (int i = 0; i < nc * n; i++) {  
        // ux[i] = ((float) rand() / RAND_MAX) * 2.0f - 1.0f; // Random float between -1 and 1  
        ux[i] = 1.0f;
    }  
  
    for (int i = 0; i < n; i++) {  
        // uy[i] = ((float) rand() / RAND_MAX) * 2.0f - 1.0f; // Random float between -1 and 1  
        uy[i] = 1.0f;
    }  
  
    // Quantize ux to vx  
    quantize_q4_0_4x4(ux, vx, nc, n);  
  
    // Quantize uy to vy  
    quantize_q8_0_4x4(uy, vy, n);  
  
    // Initialize s  
    for (int i = 0; i < nc; i++) {  
        s[i] = 0.0f;  
    }  
  
    *vx_out = vx;  
    *vy_out = vy;  
    *s_out = s;  
    *unquantized_vx = ux;  
    *unquantized_vy = uy;  
}  
  
// Benchmark function  
void benchmark_implementation(  
    const char *name,  
    void (*func)(  
        int n,  
        float * restrict s,  
        size_t bs,  
        const void * restrict vx,  
        const void * restrict vy,  
        int nr,  
        int nc  
    ),  
    int iterations,  
    int n,  
    int nc  
) {  
    block_q4_0x4 *vx;  
    block_q8_0 *vy;  
    float *s;  
    float *ux;  
    float *uy;  
    generate_test_data(n, nc, &vx, &vy, &s, &ux, &uy);  
  
    float *s_quantized = (float *) malloc(nc * sizeof(float));  
    float *s_unquantized = (float *) malloc(nc * sizeof(float));  
  
    // Warmup  
    for (int i = 0; i < 5; i++) {  
        func(n, s_quantized, QK8_0, vx, vy, 1, nc);  
    }  
  
    // Benchmark quantized GEMV  
    struct timespec start_q, end_q;  
    clock_gettime(CLOCK_MONOTONIC, &start_q);  
    for (int i = 0; i < iterations; i++) {  
        func(n, s_quantized, QK8_0, vx, vy, 1, nc);  
    }  
    clock_gettime(CLOCK_MONOTONIC, &end_q);  
  
    // Benchmark unquantized GEMV  
    struct timespec start_uq, end_uq;  
    clock_gettime(CLOCK_MONOTONIC, &start_uq);  
    for (int i = 0; i < iterations; i++) {  
        gemv_unquantized(n, s_unquantized, ux, uy, 1, nc);  
    }  
    clock_gettime(CLOCK_MONOTONIC, &end_uq);  
  
    // Calculate duration in seconds  
    double duration_q = (end_q.tv_sec - start_q.tv_sec) + (end_q.tv_nsec - start_q.tv_nsec) / 1e9;  
    double duration_uq = (end_uq.tv_sec - start_uq.tv_sec) + (end_uq.tv_nsec - start_uq.tv_nsec) / 1e9;  
  
    printf("%s (Quantized): %.2f ms per iteration\n",  
           name,  
           (duration_q * 1000.0) / iterations);  
  
    printf("%s (Unquantized): %.2f ms per iteration\n",  
           "Unquantized GEMV",  
           (duration_uq * 1000.0) / iterations);  
  
    // Compare outputs  
    printf("Comparing outputs between quantized and unquantized GEMV:\n");  
    double max_diff = 0.0;  
    for (int i = 0; i < nc; i++) {  
        double diff = fabs(s_quantized[i] - s_unquantized[i]);  
        if (diff > max_diff) {  
            max_diff = diff;  
        }  
        // Optionally, print the differences  
        // if (diff == max_diff) {
        //     printf("Index %d: Quantized = %.6f, Unquantized = %.6f, Diff = %.6f\n",  
        //        i, s_quantized[i], s_unquantized[i], diff);  
        // }
    }  
    printf("Maximum difference: %.6f\n", max_diff);  

    // Print the first 10 elements of each output
    printf("First 10 elements of quantized output:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", s_quantized[i]);
    }
    printf("\n");

    printf("First 10 elements of unquantized output:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", s_unquantized[i]);
    }
    printf("\n");
  
    free(vx);  
    free(vy);  
    free(s);  
    free(ux);  
    free(uy);  
    free(s_quantized);  
    free(s_unquantized);  
}  
  
int main() {  
    int n = 4096 * 2;  
    int nc = 4096 * 2;  
    int iterations = 1;  
  
    printf("Benchmarking with n=%d, nc=%d, iterations=%d\n", n, nc, iterations);  
  
    // Benchmark scalar implementation (Quantized GEMV)  
    benchmark_implementation(  
        "Quantized GEMV scalar",  
        (void (*)(int, float *, size_t, const void *, const void *, int, int)) gemv_q4_0_4x4_q8_0_scalar,  
        iterations,  
        n,  
        nc  
    );  

    // Benchmark scalar implementation (Quantized GEMV)  
    benchmark_implementation(  
        "Quantized GEMV",  
        (void (*)(int, float *, size_t, const void *, const void *, int, int)) ggml_gemv_q4_0_4x4_q8_0,  
        iterations,  
        n,  
        nc  
    );  
  
    return 0;  
}  