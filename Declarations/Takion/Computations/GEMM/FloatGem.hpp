// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_FLOATGEMM_HPP
#define TAKION_COMPUTE_FLOATGEMM_HPP

#include <Takion/Computations/GEMM/Gemm.hpp>
#include <xmmintrin.h>
#include <immintrin.h>
#include <algorithm>

namespace Takion::Compute::CPU
{
template <>
inline void MultiplyCpu(const Span<float> inputA,
                        const Span<float> inputB,
                        Span<float> out, std::size_t numRowA,
                        std::size_t numColA,
                        std::size_t numRowB, std::size_t numColB,
                        std::size_t numMatrices)
{
    const auto jb = std::min(static_cast<std::size_t>(512), numColB);
    const auto kb = std::min(static_cast<std::size_t>(24), numRowB);
    const auto sizeA = numRowA * numColA;
    const auto sizeB = numRowB * numColB;
    const auto sizeDest = numRowA * numColB;

#pragma omp parallel for schedule(static) default(shared)
    for (std::size_t matIdx = 0; matIdx < numMatrices; ++matIdx)
    {
        const auto matOffsetA = sizeA * matIdx;
        const auto matOffsetB = sizeB * matIdx;
        const auto matOffsetDest = sizeDest * matIdx;
        for (std::size_t jj = 0; jj < numColB; jj += jb)
        {
            for (std::size_t kk = 0; kk < numRowB; kk += kb)
            {
                for (std::size_t i = 0; i < numRowA; i += 1)
                {
                    for (std::size_t j = jj; j < std::min(jj + jb, numColB);
                         j += 8)
                    {
                        __m256 sum;
                        if (kk == 0)
                            sum = _mm256_setzero_ps();
                        else
                        {
                            sum = _mm256_load_ps(static_cast<float const*>(
                                &out[matOffsetDest + i * numColB + j]));
                        }
                        const auto limit = std::min(
                            static_cast<std::size_t>(numRowB), kk + kb);
                        for (std::size_t k = kk; k < limit; k++)
                        {
                            const auto input_a_offset =
                                matOffsetA + i * numColA + k;
                            const auto input_b_offset =
                                matOffsetB + k * numColB + j;

                            const auto bc_mat1_1 = _mm256_set1_ps(
                                inputA[input_a_offset]);
                            const auto vecA_mat2 =
                                _mm256_load_ps(static_cast<float const*>(
                                    &inputB[input_b_offset]));

                            sum = _mm256_add_ps(
                                sum, _mm256_mul_ps(bc_mat1_1, vecA_mat2));
                        }
                        _mm256_store_ps(
                            static_cast<float*>(
                                &out[matOffsetDest + i * numColB + j]),
                            sum);
                    }
                }
            }
        }
    }
}

template <>
inline void MultiplyWithBroadcastCpu(const Span<float> inputA,
                                     const Span<float> inputB, Span<float> out,
                                     std::size_t numRowA, std::size_t numColA,
                                     std::size_t numRowB, std::size_t numColB,
                                     std::size_t numMatrices, bool broadCastA)
{
    const auto jb = std::min(static_cast<std::size_t>(512), numColB);
    const auto kb = std::min(static_cast<std::size_t>(24), numRowB);
    const auto sizeA = numRowA * numColA;
    const auto sizeB = numRowB * numColB;
    const auto sizeDest = numRowA * numColB;

#pragma omp parallel for schedule(static) default(shared)
    for (std::size_t matIdx = 0; matIdx < numMatrices; ++matIdx)
    {
        const auto batchOffsetA = broadCastA ? 0 : sizeA * matIdx;
        const auto batchOffsetB = !broadCastA ? 0 : sizeB * matIdx;
        const auto batchOffsetDest = sizeDest * matIdx;
        for (std::size_t jj = 0; jj < numColB; jj += jb)
        {
            for (std::size_t kk = 0; kk < numRowB; kk += kb)
            {
                for (std::size_t i = 0; i < numRowA; i += 1)
                {
                    for (std::size_t j = jj; j < std::min(jj + jb, numColB);
                         j += 8)
                    {
                        __m256 sum;
                        if (kk == 0)
                            sum = _mm256_setzero_ps();
                        else
                        {
                            sum = _mm256_load_ps(static_cast<float const*>(
                                &out[batchOffsetDest + i * numColB + j]));
                        }
                        const auto limit = std::min(
                            static_cast<std::size_t>(numRowB), kk + kb);
                        for (std::size_t k = kk; k < limit; k++)
                        {
                            const auto input_a_offset =
                                batchOffsetA + i * numColA + k;
                            const auto input_b_offset =
                                batchOffsetB + k * numColB + j;

                            const auto bc_mat1_1 =
                                _mm256_set1_ps(inputA[input_a_offset]);
                            const auto vecA_mat2 =
                                _mm256_load_ps(static_cast<float const*>(
                                    &inputB[input_b_offset]));

                            sum = _mm256_add_ps(
                                sum, _mm256_mul_ps(bc_mat1_1, vecA_mat2));
                        }
                        _mm256_store_ps(
                            static_cast<float*>(
                                &out[batchOffsetDest + i * numColB + j]),
                            sum);
                    }
                }
            }
        }
    }
}

template <>
inline void CpuTranspose(const Span<float> in, Span<float> out,
                         std::size_t numRowInput,
                         std::size_t numColInput, std::size_t batchSize)
{
    const auto blockSize = 4;
    const auto matrixSize = numRowInput * numColInput;
    //! Optimized matrix transpose minimizing cache misses
#pragma omp parallel for schedule(static) default(shared)
    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        auto batchOffset = matrixSize * batchIdx;
        for (std::size_t ii = 0; ii < numRowInput; ii += blockSize)
            for (std::size_t jj = 0; jj < numColInput; jj += blockSize)
            {
                std::size_t i_lim = ii + blockSize;
                if (i_lim > numRowInput)
                    i_lim = numRowInput;

                std::size_t j_lim = jj + blockSize;
                if (j_lim > numColInput)
                    j_lim = numColInput;

                auto inputIndex = batchOffset + ii * numColInput + jj;
                auto outputIndex = batchOffset + jj * numRowInput + ii;

                __m128 row1 = _mm_load_ps(&in[inputIndex + blockSize * 0]);
                __m128 row2 = _mm_load_ps(&in[inputIndex + blockSize * 1]);
                __m128 row3 = _mm_load_ps(&in[inputIndex + blockSize * 2]);
                __m128 row4 = _mm_load_ps(&in[inputIndex + blockSize * 3]);
                _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
                _mm_store_ps(&out[outputIndex + blockSize * 0], row1);
                _mm_store_ps(&out[outputIndex + blockSize * 1], row2);
                _mm_store_ps(&out[outputIndex + blockSize * 2], row3);
                _mm_store_ps(&out[outputIndex + blockSize * 3], row4);
            }
    }
}

template <>
inline void ShrinkCpu(const Span<float> input, Span<float> output,
                      std::size_t size,
                      std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecA1 = _mm256_load_ps(
                static_cast<float const*>(&input[batchOffset + i]));
            const auto vecA2 = _mm256_load_ps(
                static_cast<float const*>(&input[batchOffset + i + 8]));

            const auto vecB1 = _mm256_load_ps(
                static_cast<float const*>(&output[i]));
            const auto vecB2 = _mm256_load_ps(
                static_cast<float const*>(&output[i + 8]));

            const auto sum1 = _mm256_add_ps(vecA1, vecB1);
            const auto sum2 = _mm256_add_ps(vecA2, vecB2);

#pragma omp critical
            {
                _mm256_store_ps(static_cast<float*>(&output[i]), sum1);
                _mm256_store_ps(static_cast<float*>(&output[i + 8]), sum2);
            }
        }
    }

#pragma omp parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecMul = _mm256_set1_ps(static_cast<float>(batchSize));
            const auto vecA1 = _mm256_load_ps(
                static_cast<float const*>(&input[batchOffset + i]));
            const auto vecA2 = _mm256_load_ps(
                static_cast<float const*>(&input[batchOffset + i + 8]));

            const auto div1 = _mm256_div_ps(vecA1, vecMul);
            const auto div2 = _mm256_div_ps(vecA2, vecMul);

            _mm256_store_ps(static_cast<float*>(&output[batchOffset + i]),
                            div1);
            _mm256_store_ps(static_cast<float*>(&output[batchOffset + i + 8]),
                            div2);
        }
    }
}

template <>
inline void AddCpu(const Span<float> inputA, const Span<float> inputB,
                   Span<float> out, std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecA1 = _mm256_load_ps(
                static_cast<float const*>(&inputA[batchOffset + i]));
            const auto vecA2 = _mm256_load_ps(
                static_cast<float const*>(&inputA[batchOffset + i + 8]));

            const auto vecB1 = _mm256_load_ps(
                static_cast<float const*>(&inputB[batchOffset + i]));
            const auto vecB2 = _mm256_load_ps(
                static_cast<float const*>(&inputB[batchOffset + i + 8]));

            const auto sum1 = _mm256_add_ps(vecA1, vecB1);
            const auto sum2 = _mm256_add_ps(vecA2, vecB2);

            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i]), sum1);
            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i + 8]),
                            sum2);
        }
    }
}

template <>
inline void SubCpu(const Span<float> A, const Span<float> B, Span<float> out,
                   std::size_t size,
                   std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecA1 = _mm256_load_ps(
                static_cast<float const*>(&A[batchOffset + i]));
            const auto vecA2 = _mm256_load_ps(
                static_cast<float const*>(&A[batchOffset + i + 8]));

            const auto vecB1 = _mm256_load_ps(
                static_cast<float const*>(&B[batchOffset + i]));
            const auto vecB2 = _mm256_load_ps(
                static_cast<float const*>(&B[batchOffset + i + 8]));

            const auto sum1 = _mm256_sub_ps(vecA1, vecB1);
            const auto sum2 = _mm256_sub_ps(vecA2, vecB2);

            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i]), sum1);
            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i + 8]),
                            sum2);
        }
    }
}

template <>
inline void AddWithBroadcastCpu(const Span<float> A, const Span<float> B,
                                Span<float> out,
                                std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecA1 = _mm256_load_ps(
                static_cast<float const*>(&A[batchOffset + i]));
            const auto vecA2 = _mm256_load_ps(
                static_cast<float const*>(&A[batchOffset + i + 8]));

            const auto vecB1 = _mm256_load_ps(
                static_cast<float const*>(&B[i]));
            const auto vecB2 = _mm256_load_ps(
                static_cast<float const*>(&B[i + 8]));

            const auto sum1 = _mm256_add_ps(vecA1, vecB1);
            const auto sum2 = _mm256_add_ps(vecA2, vecB2);

            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i]), sum1);
            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i + 8]),
                            sum2);
        }
    }
}

template <>
inline void DotCpu(const Span<float> inputA, const Span<float> inputB,
                   Span<float> out, std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecA1 = _mm256_load_ps(
                static_cast<float const*>(&inputA[batchOffset + i]));
            const auto vecA2 = _mm256_load_ps(
                static_cast<float const*>(&inputA[batchOffset + i + 8]));

            const auto vecB1 = _mm256_load_ps(
                static_cast<float const*>(&inputB[batchOffset + i]));
            const auto vecB2 = _mm256_load_ps(
                static_cast<float const*>(&inputB[batchOffset + i + 8]));

            const auto mul1 = _mm256_mul_ps(vecA1, vecB1);
            const auto mul2 = _mm256_mul_ps(vecA2, vecB2);

            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i]), mul1);
            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i + 8]),
                            mul2);
        }
    }
}

template <>
inline void ScalarMulCpu(const Span<float> input, float toMul, Span<float> out,
                         std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecMul = _mm256_set1_ps(toMul);
            const auto vecA1 = _mm256_load_ps(
                static_cast<float const*>(&input[batchOffset + i]));
            const auto vecA2 = _mm256_load_ps(
                static_cast<float const*>(&input[batchOffset + i + 8]));

            const auto mul1 = _mm256_mul_ps(vecA1, vecMul);
            const auto mul2 = _mm256_mul_ps(vecA2, vecMul);

            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i]), mul1);
            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i + 8]),
                            mul2);
        }
    }
}

template <>
inline void ScalarDivCpu(const Span<float> input, float toDiv, Span<float> out,
                         std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecMul = _mm256_set1_ps(toDiv);
            const auto vecA1 = _mm256_load_ps(
                static_cast<float const*>(&input[batchOffset + i]));
            const auto vecA2 = _mm256_load_ps(
                static_cast<float const*>(&input[batchOffset + i + 8]));

            const auto mul1 = _mm256_div_ps(vecA1, vecMul);
            const auto mul2 = _mm256_div_ps(vecA2, vecMul);

            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i]), mul1);
            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i + 8]),
                            mul2);
        }
    }
}

template <>
inline void SetCpu(Span<float> data, float toSet, std::size_t size,
                   std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto zero = _mm256_set1_ps(toSet);
            _mm256_store_ps(&data[batchOffset + i], zero);
        }
    }
}
}

#endif
