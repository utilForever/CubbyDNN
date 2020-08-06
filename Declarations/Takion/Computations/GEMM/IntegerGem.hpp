// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_INTEGERGEMM_HPP
#define TAKION_COMPUTE_INTEGERGEMM_HPP

#include <Takion/Computations/GEMM/Gemm.hpp>
#include <immintrin.h>
#include <algorithm>

namespace Takion::Compute::CPU
{
template <>
inline void MultiplyCpu(const Span<int> inputA, const Span<int> inputB,
                        Span<int> out, unsigned numRow, unsigned numCol,
                        unsigned numMiddle, unsigned batchSize)
{
    const auto jb = std::min(512u, numCol);
    const auto kb = std::min(24u, numRow);
    const auto sizeA = numRow * numMiddle;
    const auto sizeB = numMiddle * numCol;
    const auto sizeDest = numRow * numCol;

#pragma parallel for schedule(static) default(shared)
    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        const auto batchOffsetA = sizeA * batchIdx;
        const auto batchOffsetB = sizeB * batchIdx;
        const auto batchOffsetDest = sizeDest * batchIdx;
        for (std::size_t jj = 0; jj < numCol; jj += jb)
        {
            for (std::size_t kk = 0; kk < numMiddle; kk += kb)
            {
                for (std::size_t i = 0; i < numRow; i += 1)
                {
                    for (std::size_t j = jj; j < jj + jb; j += 16)
                    {
                        __m256i sumA_1, sumB_1;
                        if (kk == 0)
                        {
                            sumA_1 = sumB_1 = _mm256_setzero_si256();
                        }
                        else
                        {
                            sumA_1 = _mm256_load_si256(
                                (__m256i*)(&out[batchOffsetDest + i * numCol +
                                                j]));
                            sumB_1 = _mm256_load_si256(
                                (__m256i*)&out[batchOffsetDest + i * numCol +
                                               j + 8]);
                        }
                        const auto limit = std::min(
                            static_cast<std::size_t>(numMiddle), kk + kb);
                        for (size_t k = kk; k < limit; k++)
                        {
                            auto bc_mat1_1 = _mm256_set1_epi32(
                                inputA[batchOffsetA + i * numMiddle + k]);
                            auto vecA_mat2 = _mm256_loadu_si256(
                                (__m256i*)&inputB[batchOffsetB + k * numRow +
                                                  j]);
                            auto vecB_mat2 = _mm256_loadu_si256(
                                (__m256i*)&inputB[batchOffsetB + k * numRow +
                                                  j + 8]);
                            sumA_1 = _mm256_add_epi32(
                                sumA_1,
                                _mm256_mullo_epi32(bc_mat1_1, vecA_mat2));
                            sumB_1 = _mm256_add_epi32(
                                sumB_1,
                                _mm256_mullo_epi32(bc_mat1_1, vecB_mat2));
                        }
                        _mm256_storeu_si256(
                            (__m256i*)&out[batchOffsetDest + i * numCol + j],
                            sumA_1);
                        _mm256_storeu_si256((__m256i*)&out[batchOffsetDest +
                                                           i * numCol + j + 8],
                                            sumB_1);
                    }
                }
            }
        }
    }
}

template <>
inline void MultiplyWithBroadcastCpu(const Span<int> inputA,
                                     const Span<int> inputB,
                                     Span<int> out, unsigned numRow,
                                     unsigned numCol,
                                     unsigned numMiddle, unsigned batchSize,
                                     bool broadCastA)
{
    const auto jb = std::min(512u, numCol);
    const auto kb = std::min(24u, numRow);
    const auto sizeA = numRow * numMiddle;
    const auto sizeB = numMiddle * numCol;
    const auto sizeDest = numRow * numCol;

#pragma parallel for schedule(static) default(shared)
    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        const auto batchOffsetA = broadCastA ? 0 : sizeA * batchIdx;
        const auto batchOffsetB = !broadCastA ? 0 : sizeB * batchIdx;
        const auto batchOffsetDest = sizeDest * batchIdx;
        for (std::size_t jj = 0; jj < numCol; jj += jb)
        {
            for (std::size_t kk = 0; kk < numMiddle; kk += kb)
            {
                for (std::size_t i = 0; i < numRow; i += 1)
                {
                    for (std::size_t j = jj; j < jj + jb; j += 16)
                    {
                        __m256i sumA, sumB;
                        if (kk == 0)
                        {
                            sumA = sumB = _mm256_setzero_si256();
                        }
                        else
                        {
                            sumA = _mm256_load_si256(
                                (__m256i*)&out[batchOffsetDest + i * numCol +
                                               j]);
                            sumB = _mm256_load_si256(
                                (__m256i*)&out[batchOffsetDest + i * numCol +
                                               j + 8]);
                        }
                        const auto limit = std::min(
                            static_cast<std::size_t>(numMiddle), kk + kb);
                        for (size_t k = kk; k < limit; k++)
                        {
                            auto bc_mat1_1 = _mm256_set1_epi32(
                                inputA[batchOffsetA + i * numMiddle + k]);
                            auto vecA_mat2 = _mm256_loadu_si256(
                                (__m256i*)&inputB[batchOffsetB + k * numRow +
                                                  j]);
                            auto vecB_mat2 = _mm256_loadu_si256(
                                (__m256i*)&inputB[batchOffsetB + k * numRow +
                                                  j + 8]);
                            sumA = _mm256_add_epi32(
                                sumA,
                                _mm256_mullo_epi32(bc_mat1_1, vecA_mat2));
                            sumB = _mm256_add_epi32(
                                sumB,
                                _mm256_mullo_epi32(bc_mat1_1, vecB_mat2));
                        }
                        _mm256_storeu_si256(
                            (__m256i*)&out[batchOffsetDest + i * numCol + j],
                            sumA);
                        _mm256_storeu_si256((__m256i*)&out[batchOffsetDest +
                                                           i * numCol + j + 8],
                                            sumB);
                    }
                }
            }
        }
    }
}

template <>
inline void CpuTranspose(const Span<int> input, Span<int> output,
                         unsigned numRowInput,
                         unsigned numColInput, unsigned batchSize)
{
    const auto blockSize = 4;
    const auto matrixSize = numRowInput * numColInput;
    //! Optimized matrix transpose minimizing cache misses
    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t ii = 0; ii < numRowInput; ii += blockSize)
            for (std::size_t jj = 0; jj < numColInput; jj += blockSize)
            {
                std::size_t i_lim = ii + blockSize;
                if (i_lim > numRowInput)
                    i_lim = numRowInput;

                std::size_t j_lim = jj + blockSize;
                if (j_lim > numColInput)
                    j_lim = numColInput;

                for (std::size_t i = ii; i < i_lim; i++)
                    for (std::size_t j = jj; j < j_lim; j++)
                        output[batchIdx * matrixSize +
                               j * numRowInput + i] =
                            input[batchIdx * matrixSize +
                                  i * numColInput + j];
            }
    }
}

template <>
void ShrinkCpu(const Span<int> input, Span<int> output, unsigned size,
               unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecA1 =
                _mm256_loadu_si256((__m256i*)&input[batchOffset + i]);
            const auto vecA2 =
                _mm256_loadu_si256((__m256i*)&input[batchOffset + i + 8]);

            const auto vecB1 =
                _mm256_loadu_si256((__m256i*)&output[i]);
            const auto vecB2 =
                _mm256_loadu_si256((__m256i*)&output[i + 8]);

            const auto sum1 = _mm256_add_epi32(vecA1, vecB1);
            const auto sum2 = _mm256_add_epi32(vecA2, vecB2);

            _mm256_storeu_si256((__m256i*)&output[i], sum1);
            _mm256_storeu_si256((__m256i*)&output[i + 8], sum2);
        }
    }
}


template <>
inline void AddCpu(const Span<int> A, const Span<int> B,
                   Span<int> out,
                   unsigned size, unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecA1 =
                _mm256_loadu_si256((__m256i*)&A[batchOffset + i]);
            const auto vecA2 =
                _mm256_loadu_si256((__m256i*)&A[batchOffset + i + 8]);

            const auto vecB1 =
                _mm256_loadu_si256((__m256i*)&B[batchOffset + i]);
            const auto vecB2 =
                _mm256_loadu_si256((__m256i*)&B[batchOffset + i + 8]);

            const auto sum1 = _mm256_add_epi32(vecA1, vecB1);
            const auto sum2 = _mm256_add_epi32(vecA2, vecB2);

            _mm256_storeu_si256((__m256i*)&out[batchOffset + i], sum1);
            _mm256_storeu_si256((__m256i*)&out[batchOffset + i + 8], sum2);
        }
    }
}

template <>
inline void AddWithBroadcastCpu(const Span<int> A, const Span<int> B,
                         Span<int> out, unsigned size, unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecA1 =
                _mm256_loadu_si256((__m256i*)&A[batchOffset + i]);
            const auto vecA2 =
                _mm256_loadu_si256((__m256i*)&A[batchOffset + i + 8]);

            const auto vecB1 =
                _mm256_loadu_si256((__m256i*)&B[i]);
            const auto vecB2 =
                _mm256_loadu_si256((__m256i*)&B[i + 8]);

            const auto sum1 = _mm256_add_epi32(vecA1, vecB1);
            const auto sum2 = _mm256_add_epi32(vecA2, vecB2);

            _mm256_storeu_si256((__m256i*)&out[batchOffset + i], sum1);
            _mm256_storeu_si256((__m256i*)&out[batchOffset + i + 8], sum2);
        }
    }
}

template <>
inline void DotCpu(const Span<int> inputA, const Span<int> inputB,
                   Span<int> out,
                   unsigned size, unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecA1 =
                _mm256_loadu_si256((__m256i*)&inputA[batchOffset + i]);
            const auto vecA2 =
                _mm256_loadu_si256((__m256i*)&inputA[batchOffset + i + 8]);

            const auto vecB1 =
                _mm256_loadu_si256((__m256i*)&inputB[batchOffset + i]);
            const auto vecB2 =
                _mm256_loadu_si256((__m256i*)&inputB[batchOffset + i + 8]);

            const auto mul1 = _mm256_mullo_epi32(vecA1, vecB1);
            const auto mul2 = _mm256_mullo_epi32(vecA2, vecB2);

            _mm256_storeu_si256((__m256i*)&out[batchOffset + i], mul1);
            _mm256_storeu_si256((__m256i*)&out[batchOffset + i + 8], mul2);
        }
    }
}


template <>
inline void ScalarMulCpu(const Span<int> input, int toMul, Span<int> out,
                         unsigned size, unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecMul = _mm256_set1_epi32(toMul);
            const auto vecA1 =
                _mm256_loadu_si256((__m256i*)&input[batchOffset + i]);
            const auto vecA2 =
                _mm256_loadu_si256((__m256i*)&input[batchOffset + i + 8]);

            const auto mul1 = _mm256_mullo_epi32(vecA1, vecMul);
            const auto mul2 = _mm256_mullo_epi32(vecA2, vecMul);

            _mm256_storeu_si256((__m256i*)&out[batchOffset + i], mul1);
            _mm256_storeu_si256((__m256i*)&out[batchOffset + i + 8], mul2);
        }
    }
}

template<>
inline void ScalarDivCpu(const Span<int> input, int toDiv, Span<int> out,
                         unsigned size, unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecMul = _mm256_set1_epi32(toDiv);
            const auto vecA1 =
                _mm256_loadu_si256((__m256i*)&input[batchOffset + i]);
            const auto vecA2 =
                _mm256_loadu_si256((__m256i*)&input[batchOffset + i + 8]);

            const auto mul1 = _mm256_div_epi32(vecA1, vecMul);
            const auto mul2 = _mm256_div_epi32(vecA2, vecMul);

            _mm256_storeu_si256((__m256i*)&out[batchOffset + i], mul1);
            _mm256_storeu_si256((__m256i*)&out[batchOffset + i + 8], mul2);
        }
    }
}
}

#endif
