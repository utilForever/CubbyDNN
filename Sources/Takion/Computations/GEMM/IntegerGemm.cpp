// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Takion/Computations/GEMM/IntegerGemm.hpp>
#include <Takion/Utils/Span.hpp>
#include <immintrin.h>
#include <algorithm>
#include <iostream>

namespace Takion::Compute::CPU::Int
{
void MultiplyCpu(const Span<int> inputA, const Span<int> inputB,
                 Span<int> out, std::size_t numRowA, std::size_t numColA,
                 std::size_t numRowB, std::size_t numColB,
                 std::size_t numMatrices)
{
    const auto jb = std::min(static_cast<std::size_t>(512), numColB);
    const auto kb = std::min(static_cast<std::size_t>(24), numRowB);
    const auto sizeA = numRowA * numColA;
    const auto sizeB = numRowB * numColB;
    const auto sizeDest = numRowA * numColB;

#pragma omp parallel for schedule(static) default(shared)
    for (long matIdx = 0; static_cast<std::size_t>(matIdx) < numMatrices;
         ++matIdx)
    {
        const auto batchOffsetA = sizeA * matIdx;
        const auto batchOffsetB = sizeB * matIdx;
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
                        __m256i sum;
                        if (kk == 0)
                            sum = _mm256_setzero_si256();
                        else
                        {
                            sum = _mm256_load_si256(
                                (__m256i*)&out[batchOffsetDest + i * numColB +
                                               j]);
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
                                _mm256_set1_epi32(inputA[input_a_offset]);
                            const auto vecA_mat2 = _mm256_load_si256(
                                (__m256i*)&inputB[input_b_offset]);

                            sum = _mm256_add_epi32(
                                sum, _mm256_mullo_epi32(bc_mat1_1, vecA_mat2));
                        }
                        _mm256_store_si256(
                            (__m256i*)&out[batchOffsetDest + i * numColB + j],
                            sum);
                    }
                }
            }
        }
    }
}


void MultiplyWithBroadcastCpu(const Span<int> inputA,
                              const Span<int> inputB, Span<int> out,
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
    for (long matIdx = 0; static_cast<std::size_t>(matIdx) < numMatrices;
         ++matIdx)
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
                        __m256i sum;
                        if (kk == 0)
                            sum = _mm256_setzero_si256();
                        else
                        {
                            sum = _mm256_load_si256(
                                (__m256i*)&out[batchOffsetDest + i * numColB +
                                               j]);
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
                                _mm256_set1_epi32(inputA[input_a_offset]);
                            const auto vecA_mat2 = _mm256_load_si256(
                                (__m256i*)&inputB[input_b_offset]);

                            sum = _mm256_add_epi32(
                                sum, _mm256_mullo_epi32(bc_mat1_1, vecA_mat2));
                        }
                        _mm256_store_si256(
                            (__m256i*)&out[batchOffsetDest + i * numColB + j],
                            sum);
                    }
                }
            }
        }
    }
}


void CpuTranspose(const Span<int> input, Span<int> output,
                  std::size_t numRowInput, std::size_t numColInput,
                  std::size_t batchSize)
{
    const auto blockSize = 4;
    const auto matrixSize = numRowInput * numColInput;
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         ++batchIdx)
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
                        output[batchIdx * matrixSize + j * numRowInput + i] =
                            input[batchIdx * matrixSize + i * numColInput + j];
            }
    }
}


void ShrinkCpu(const Span<int> input, Span<int> output, std::size_t size,
               std::size_t batchSize)
{
    // #pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecA =
                _mm256_loadu_si256((__m256i*)&input[batchOffset + i]);
            const auto vecB = _mm256_loadu_si256((__m256i*)&output[i]);
            const auto sum = _mm256_add_epi32(vecA, vecB);
            // #pragma omp critical
            {
                _mm256_storeu_si256((__m256i*)&output[i], sum);
            }
        }
    }

#ifdef _MSC_VER
#if _MSC_VER >= 1920
#pragma omp parallel for schedule(static) default(shared)
    for (long i = 0; static_cast<std::size_t>(i) < size; i += 8)
    {
        const auto vecDiv = _mm256_set1_epi32(static_cast<int>(batchSize));
        const auto vec = _mm256_loadu_si256((__m256i*)&output[i]);
        const auto div = _mm256_div_epi32(vec, vecDiv);
        _mm256_storeu_si256((__m256i*)&output[i], div);
    }
#else
#pragma omp parallel for schedule(static) default(shared)
    for (long i = 0; static_cast<std::size_t>(i) < size; i += 1)
    {
        output[i] /= static_cast<int>(batchSize);
    }
#endif
#else
#pragma omp parallel for schedule(static) default(shared)
    for (long i = 0; static_cast<std::size_t>(i) < size; i += 1)
    {
        output[i] /= static_cast<int>(batchSize);
    }
#endif
}


void AddCpu(const Span<int> A, const Span<int> B, Span<int> out,
            std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecA1 =
                _mm256_loadu_si256((__m256i*)&A[batchOffset + i]);
            const auto vecB1 =
                _mm256_loadu_si256((__m256i*)&B[batchOffset + i]);
            const auto sum1 = _mm256_add_epi32(vecA1, vecB1);
            _mm256_storeu_si256((__m256i*)&out[batchOffset + i], sum1);
        }
    }
}


void SubCpu(const Span<int> A, const Span<int> B, Span<int> out,
            std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 16)
        {
            const auto vecA1 =
                _mm256_loadu_si256((__m256i*)&A[batchOffset + i]);
            const auto vecB1 =
                _mm256_loadu_si256((__m256i*)&B[batchOffset + i]);
            const auto sum1 = _mm256_sub_epi32(vecA1, vecB1);
            _mm256_storeu_si256((__m256i*)&out[batchOffset + i], sum1);
        }
    }
}


void AddWithBroadcastCpu(const Span<int> A, const Span<int> B,
                         Span<int> out, std::size_t size,
                         std::size_t batchSize, bool broadCastA)
{
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffsetA = broadCastA ? 0 : size * batchIdx;
        const auto batchOffsetB = broadCastA ? size * batchIdx : 0;
        const auto batchOffsetOut = broadCastA ? batchOffsetB : batchOffsetA;

        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecA =
                _mm256_loadu_si256((__m256i*)&A[batchOffsetA + i]);
            const auto vecB1 =
                _mm256_loadu_si256((__m256i*)&B[batchOffsetB + i]);
            const auto sum = _mm256_add_epi32(vecA, vecB1);
            _mm256_storeu_si256((__m256i*)&out[batchOffsetOut + i], sum);
        }
    }
}


void SubWithBroadcastCpu(const Span<int> A, const Span<int> B,
                         Span<int> out, std::size_t size,
                         std::size_t batchSize, bool broadCastA)
{
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffsetA = broadCastA ? 0 : size * batchIdx;
        const auto batchOffsetB = broadCastA ? size * batchIdx : 0;
        const auto batchOffsetOut = broadCastA ? batchOffsetB : batchOffsetA;

        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecA =
                _mm256_loadu_si256((__m256i*)&A[batchOffsetA + i]);
            const auto vecB1 =
                _mm256_loadu_si256((__m256i*)&B[batchOffsetB + i]);
            const auto sum = _mm256_sub_epi32(vecA, vecB1);
            _mm256_storeu_si256((__m256i*)&out[batchOffsetOut + i], sum);
        }
    }
}


void DotCpu(const Span<int> inputA, const Span<int> inputB,
            Span<int> out, std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecA1 =
                _mm256_loadu_si256((__m256i*)&inputA[batchOffset + i]);
            const auto vecB1 =
                _mm256_loadu_si256((__m256i*)&inputB[batchOffset + i]);
            const auto mul1 = _mm256_mullo_epi32(vecA1, vecB1);
            _mm256_storeu_si256((__m256i*)&out[batchOffset + i], mul1);
        }
    }
}


void DotWithBroadcastCpu(const Span<int> inputA, const Span<int> inputB,
                         Span<int> out, std::size_t size,
                         std::size_t batchSize, bool broadCastA)
{
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffsetA = broadCastA ? 0 : size * batchIdx;
        const auto batchOffsetB = broadCastA ? size * batchIdx : 0;
        const auto batchOffsetOut = broadCastA ? batchOffsetB : batchOffsetA;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecA1 =
                _mm256_loadu_si256((__m256i*)&inputA[batchOffsetA + i]);
            const auto vecB1 =
                _mm256_loadu_si256((__m256i*)&inputB[batchOffsetB + i]);
            const auto mul1 = _mm256_mullo_epi32(vecA1, vecB1);
            _mm256_storeu_si256((__m256i*)&out[batchOffsetOut + i], mul1);
        }
    }
}


void DivCpu(const Span<int> inputA, const Span<int> inputB,
            Span<int> out, std::size_t size, std::size_t batchSize)
{
#ifdef _MSC_VER
#if _MSC_VER >= 1920
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecA =
                _mm256_loadu_si256((__m256i*)&inputA[batchOffset + i]);
            const auto vecB =
                _mm256_loadu_si256((__m256i*)&inputB[batchOffset + i]);
            const auto div = _mm256_div_epi32(vecA, vecB);
            _mm256_storeu_si256((__m256i*)&out[batchOffset + i], div);
        }
    }
#else
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 1)
        {
            out[batchOffset + i] =
                inputA[batchOffset + i] / inputB[batchOffset + i];
        }
    }
#endif
#else
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 1)
        {
            out[batchOffset + i] =
                inputA[batchOffset + i] / inputB[batchOffset + i];
        }
    }
#endif
}


void DivWithBroadcastCpu(const Span<int> inputA, const Span<int> inputB,
                         Span<int> out, std::size_t size,
                         std::size_t batchSize, bool broadCastA)
{
#ifdef _MSC_VER
#if _MSC_VER >= 1920
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffsetA = broadCastA ? 0 : size * batchIdx;
        const auto batchOffsetB = broadCastA ? size * batchIdx : 0;
        const auto batchOffsetOut = broadCastA ? batchOffsetB : batchOffsetA;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecA =
                _mm256_loadu_si256((__m256i*)&inputA[batchOffsetA + i]);
            const auto vecB =
                _mm256_loadu_si256((__m256i*)&inputB[batchOffsetB + i]);
            const auto div = _mm256_div_epi32(vecA, vecB);
            _mm256_storeu_si256((__m256i*)&out[batchOffsetOut + i], div);
        }
    }
#else
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffsetA = broadCastA ? 0 : size * batchIdx;
        const auto batchOffsetB = broadCastA ? size * batchIdx : 0;
        const auto batchOffsetOut = broadCastA ? batchOffsetB : batchOffsetA;
        for (std::size_t i = 0; i < size; i += 1)
        {
            out[batchOffsetOut + i] =
                inputA[batchOffsetA + i] / inputB[batchOffsetB + i];
        }
    }
#endif
#else
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffsetA = broadCastA ? 0 : size * batchIdx;
        const auto batchOffsetB = broadCastA ? size * batchIdx : 0;
        const auto batchOffsetOut = broadCastA ? batchOffsetB : batchOffsetA;
        for (std::size_t i = 0; i < size; i += 1)
        {
            out[batchOffsetOut + i] =
                inputA[batchOffsetA + i] / inputB[batchOffsetB + i];
        }
    }
#endif
}


void ScalarMulCpu(const Span<int> input, int toMul, Span<int> out,
                  std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecMul = _mm256_set1_epi32(toMul);
            const auto vecA =
                _mm256_loadu_si256((__m256i*)&input[batchOffset + i]);
            const auto mul = _mm256_mullo_epi32(vecA, vecMul);
            _mm256_storeu_si256((__m256i*)&out[batchOffset + i], mul);
        }
    }
}


void ScalarDivCpu(const Span<int> input, int toDiv, Span<int> out,
                  std::size_t size, std::size_t batchSize)
{
#ifdef _MSC_VER
#if _MSC_VER >= 1920
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecMul = _mm256_set1_epi32(toDiv);
            const auto vecA =
                _mm256_loadu_si256((__m256i*)&input[batchOffset + i]);
            const auto mul = _mm256_div_epi32(vecA, vecMul);
            _mm256_storeu_si256((__m256i*)&out[batchOffset + i], mul);
        }
    }
#else
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 1)
        {
            out[batchOffset + i] = input[batchOffset + i] / toDiv;
        }
    }

#endif
#else
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 1)
        {
            out[batchOffset + i] = input[batchOffset + i] / toDiv;
        }
    }
#endif
}


void SetCpu(Span<int> data, int toSet, std::size_t size,
            std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto zero = _mm256_set1_epi32(toSet);
            _mm256_store_si256((__m256i*)&data[batchOffset + i], zero);
        }
    }
}
}
