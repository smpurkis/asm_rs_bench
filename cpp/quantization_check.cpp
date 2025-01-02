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

int main() {
    const int num_elements = 512; // Number of elements in the vector
    std::vector<float> random_floats(num_elements);

    // Generate random floats between -1 and 1
    for (int i = 0; i < num_elements; ++i) {
        random_floats[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    }

    // Print the first few elements of the original and q4_0 data for verification
    std::cout << "Original floats: ";
    for (int i = 0; i < num_elements; ++i) {
        if (i % QK4_0 == 0) {
            std::cout << "\n";
        }
        std::cout << random_floats[i] << " ";
    }
    std::cout << "\n";


    // Allocate memory for q8_0x4 quantized data
    std::vector<block_q8_0x4> q8_0_data(num_elements);

    // Allocate memory for q4_0_4 quantized data
    std::vector<block_q4_0> q4_0_data(num_elements / QK4_0);

    // Quantize the random floats to q4_0_4
    quantize_q4_0(random_floats.data(), q4_0_data.data(), num_elements / QK4_0, QK4_0, nullptr);

    std::cout << "Dequantized q4_0 data: ";
    for (int i = 0; i < num_elements; ++i) {
        if (i % QK4_0 == 0) {
            std::cout << "\n";
        }
        float dequantized_value = dequantize_q4_0_scalar(q4_0_data[i / QK4_0], i % QK4_0, false);
        std::cout << dequantized_value << " ";
    }
    std::cout << "\n";

    // Allocate memory for q4_0_4 quantized data
    std::vector<block_q4_0x4> q4_0_4_data(num_elements / QK4_0);

    // Quantize the random floats to q4_0_4
    quantize_q4_0_4x4(random_floats.data(), q4_0_4_data.data(), num_elements / QK4_0, QK4_0, nullptr);

    // Allocate memory for converted q4_0 data
    std::vector<block_q4_0> converted_q4_0_data(num_elements / QK4_0);

    // Convert q4_0x4 to q4_0
    convert_q4_0x4_to_q4_0(q4_0_4_data.data(), converted_q4_0_data.data(), num_elements);

    

    // Print the first few elements of the converted q4_0 data for verification
    std::cout << "Converted q4_0 data: ";
    for (int i = 0; i < num_elements; ++i) {
        if (i % QK4_0 == 0) {
            std::cout << "\n";
            std::cout << i << "\n";
        }
        float dequantized_value = dequantize_q4_0_scalar(converted_q4_0_data[i / QK4_0], i % QK4_0, false);
        std::cout << dequantized_value << " ";
    }
    std::cout << "\n";

    // Quantize the random floats to q8_0x4
    quantize_mat_q8_0(random_floats.data(), q8_0_data.data(), 4, num_elements, 4);

    std::cout << "Dequantized q8_0 data: ";
    for (int i = 0; i < num_elements; ++i) {
        if (i % QK8_0 == 0) {
            std::cout << "\n";
            std::cout << i << "\n";
        }
        float dequantized_value = dequantize_q8_0x4_scalar(q8_0_data[i / QK8_0], i % QK8_0);
        std::cout << dequantized_value << " ";
    }

    return 0;
}
