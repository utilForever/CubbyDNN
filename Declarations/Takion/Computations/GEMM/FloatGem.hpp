// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_FLOATGEM_HPP
#define TAKION_COMPUTE_FLOATGEM_HPP

#include <Takion/Computations/GEMM/Gemm.hpp>
#include <xmmintrin.h>
#include <immintrin.h>
#include <algorithm>

namespace Takion::Compute::CPU
{
template <>
inline void MultiplyCpu(const Utils::Span<float> inputA,
                        const Span<float> inputB,
                        Span<float> out, unsigned numRow, unsigned numCol,
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
                        __m256 sumA_1, sumB_1;
                        if (kk == 0)
                            sumA_1 = sumB_1 = _mm256_setzero_ps();
                        else
                        {
                            sumA_1 = _mm256_load_ps(static_cast<float const*>(
                                &out[batchOffsetDest + i * numCol + j]));
                            sumB_1 = _mm256_load_ps(static_cast<float const*>(
                                &out[batchOffsetDest + i * numCol + j + 8]));
                        }
                        const auto limit = std::min(
                            static_cast<std::size_t>(numMiddle), kk + kb);
                        for (size_t k = kk; k < limit; k++)
                        {
                            const auto bc_mat1_1 = _mm256_set1_ps(
                                inputA[batchOffsetA + i * numMiddle + k]);
                            const auto vecA_mat2 =
                                _mm256_load_ps(static_cast<float const*>(
                                    &inputB[batchOffsetB + k * numRow + j]));
                            const auto vecB_mat2 =
                                _mm256_load_ps(static_cast<float const*>(
                                    &inputB[batchOffsetB + k * numRow + j +
                                            8]));
                            sumA_1 = _mm256_add_ps(
                                sumA_1, _mm256_mul_ps(bc_mat1_1, vecA_mat2));
                            sumB_1 = _mm256_add_ps(
                                sumB_1, _mm256_mul_ps(bc_mat1_1, vecB_mat2));
                        }
                        _mm256_store_ps(
                            static_cast<float*>(
                                &out[batchOffsetDest + i * numCol + j]),
                            sumA_1);
                        _mm256_store_ps(
                            static_cast<float*>(
                                &out[batchOffsetDest + i * numCol + j + 8]),
                            sumB_1);
                    }
                }
            }
        }
    }
}

template <>
inline void MultiplyWithBroadcastCpu(const Span<float> inputA,
                                     const Span<float> inputB,
                                     Span<float> out, unsigned numRow,
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
                        __m256 sumA_1, sumB_1;
                        if (kk == 0)
                            sumA_1 = sumB_1 = _mm256_setzero_ps();
                        else
                        {
                            sumA_1 = _mm256_load_ps(static_cast<float const*>(
                                &out[batchOffsetDest + i * numCol + j]));
                            sumB_1 = _mm256_load_ps(static_cast<float const*>(
                                &out[batchOffsetDest + i * numCol + j + 8]));
                        }
                        const auto limit = std::min(
                            static_cast<std::size_t>(numMiddle), kk + kb);
                        for (std::size_t k = kk; k < limit; k++)
                        {
                            const auto bc_mat1_1 = _mm256_set1_ps(
                                inputA[batchOffsetA + i * numMiddle + k]);
                            const auto vecA_mat2 =
                                _mm256_load_ps(static_cast<float const*>(
                                    &inputB[batchOffsetB + k * numRow + j]));
                            const auto vecB_mat2 =
                                _mm256_load_ps(static_cast<float const*>(
                                    &inputB[batchOffsetB + k * numRow + j +
                                            8]));
                            sumA_1 = _mm256_add_ps(
                                sumA_1, _mm256_mul_ps(bc_mat1_1, vecA_mat2));
                            sumB_1 = _mm256_add_ps(
                                sumB_1, _mm256_mul_ps(bc_mat1_1, vecB_mat2));
                        }
                        _mm256_store_ps(
                            static_cast<float*>(
                                &out[batchOffsetDest + i * numCol + j]),
                            sumA_1);
                        _mm256_store_ps(
                            static_cast<float*>(
                                &out[batchOffsetDest + i * numCol + j + 8]),
                            sumB_1);
                    }
                }
            }
        }
    }
}

template <>
inline void CpuTranspose(const Span<float> in, Span<float> out,
                         unsigned numRowInput,
                         unsigned numColInput, unsigned batchSize)
{
    const auto blockSize = 4;
    const auto matrixSize = numRowInput * numColInput;
    //! Optimized matrix transpose minimizing cache misses
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
void ShrinkCpu(const Span<float> input, Span<float> output, unsigned size,
               unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
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

            _mm256_store_ps(static_cast<float*>(&output[i]), sum1);
            _mm256_store_ps(static_cast<float*>(&output[i + 8]),
                            sum2);
        }
    }
}

template <>
inline void AddCpu(const Span<float> inputA, const Span<float> inputB,
                   Span<float> out, unsigned size, unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
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
inline void AddWithBroadcastCpu(const Span<float> A, const Span<float> B, Span<float> out,
                         unsigned size, unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
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
                   Span<float> out, unsigned size, unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
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
                         unsigned size, unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
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
                         unsigned size, unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
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
}

#endif
