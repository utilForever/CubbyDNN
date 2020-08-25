// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Takion/Computations/GEMM/FloatGemm.hpp>
#include <Takion/Utils/Span.hpp>
#include <immintrin.h>
#include <xmmintrin.h>
#include <algorithm>
#include <iostream>

namespace Takion::Compute::CPU::Float
{
using namespace Util;

void MultiplyCpu(const Span<float> inputA, const Span<float> inputB,
                 Span<float> out, std::size_t numRowA,
                 std::size_t numColA, std::size_t numRowB,
                 std::size_t numColB, std::size_t numMatrices)
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
                                &out[matOffsetDest + i * numColB + j]),
                            sum);
                    }
                }
            }
        }
    }
}

void MultiplyWithBroadcastCpu(const Span<float> inputA,
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

void ShrinkCpu(const Span<float> input, Span<float> output,
               std::size_t size, std::size_t batchSize)
{
    //#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecA = _mm256_load_ps(
                static_cast<float const*>(&input[batchOffset + i]));
            const auto vecB =
                _mm256_load_ps(static_cast<float const*>(&output[i]));
            const auto sum = _mm256_add_ps(vecA, vecB);

            //#pragma omp critical
            {
                _mm256_store_ps(static_cast<float*>(&output[i]), sum);
            }
        }
    }

#pragma omp parallel for schedule(static) default(shared)
    for (long i = 0; static_cast<std::size_t>(i) < size; i += 8)
    {
        const auto vecDiv = _mm256_set1_ps(static_cast<float>(batchSize));
        const auto vecA = _mm256_load_ps(static_cast<float const*>(&output[i]));
        const auto div = _mm256_div_ps(vecA, vecDiv);
        _mm256_store_ps(static_cast<float*>(&output[i]), div);
    }
}

void AddCpu(const Span<float> inputA, const Span<float> inputB,
            Span<float> out, std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecA1 = _mm256_load_ps(
                static_cast<float const*>(&inputA[batchOffset + i]));
            const auto vecB1 = _mm256_load_ps(
                static_cast<float const*>(&inputB[batchOffset + i]));
            const auto sum = _mm256_add_ps(vecA1, vecB1);
            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i]), sum);
        }
    }
}

void SubCpu(const Span<float> A, const Span<float> B, Span<float> out,
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
                _mm256_load_ps(static_cast<float const*>(&A[batchOffset + i]));
            const auto vecB1 =
                _mm256_load_ps(static_cast<float const*>(&B[batchOffset + i]));
            const auto sum1 = _mm256_sub_ps(vecA1, vecB1);
            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i]), sum1);
        }
    }
}

void AddWithBroadcastCpu(const Span<float> A, const Span<float> B,
                         Span<float> out, std::size_t size,
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
                _mm256_load_ps(static_cast<float const*>(&A[batchOffsetA + i]));
            const auto vecB =
                _mm256_load_ps(static_cast<float const*>(&B[batchOffsetB + i]));
            const auto sum = _mm256_add_ps(vecA, vecB);
            _mm256_store_ps(static_cast<float*>(&out[batchOffsetOut + i]), sum);
        }
    }
}

void SubWithBroadcastCpu(const Span<float> A, const Span<float> B,
                         Span<float> out, std::size_t size,
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
                _mm256_load_ps(static_cast<float const*>(&A[batchOffsetA + i]));
            const auto vecB =
                _mm256_load_ps(static_cast<float const*>(&B[batchOffsetB + i]));
            const auto sum = _mm256_sub_ps(vecA, vecB);
            _mm256_store_ps(static_cast<float*>(&out[batchOffsetOut + i]), sum);
        }
    }
}

void DotCpu(const Span<float> inputA, const Span<float> inputB,
            Span<float> out, std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecA = _mm256_load_ps(
                static_cast<float const*>(&inputA[batchOffset + i]));
            const auto vecB = _mm256_load_ps(
                static_cast<float const*>(&inputB[batchOffset + i]));
            const auto mul = _mm256_mul_ps(vecA, vecB);
            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i]), mul);
        }
    }
}

void DotWithBroadcastCpu(const Span<float> inputA,
                         const Span<float> inputB, Span<float> out,
                         std::size_t size, std::size_t batchSize,
                         bool broadCastA)
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
            const auto vecA = _mm256_load_ps(
                static_cast<float const*>(&inputA[batchOffsetA + i]));
            const auto vecB = _mm256_load_ps(
                static_cast<float const*>(&inputB[batchOffsetB + i]));
            const auto mul = _mm256_mul_ps(vecA, vecB);
            _mm256_store_ps(static_cast<float*>(&out[batchOffsetOut + i]), mul);
        }
    }
}

void DivCpu(const Span<float> inputA, const Span<float> inputB,
            Span<float> out, std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecA = _mm256_load_ps(
                static_cast<float const*>(&inputA[batchOffset + i]));
            const auto vecB = _mm256_load_ps(
                static_cast<float const*>(&inputB[batchOffset + i]));
            const auto mul = _mm256_div_ps(vecA, vecB);
            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i]), mul);
        }
    }
}

void DivWithBroadcastCpu(const Span<float> inputA,
                         const Span<float> inputB, Span<float> out,
                         std::size_t size, std::size_t batchSize,
                         bool broadCastA)
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
            const auto vecA = _mm256_load_ps(
                static_cast<float const*>(&inputA[batchOffsetA + i]));
            const auto vecB = _mm256_load_ps(
                static_cast<float const*>(&inputB[batchOffsetB + i]));
            const auto div = _mm256_div_ps(vecA, vecB);
            _mm256_store_ps(static_cast<float*>(&out[batchOffsetOut + i]), div);
        }
    }
}

void ScalarMulCpu(const Span<float> input, float toMul, Span<float> out,
                  std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecMul = _mm256_set1_ps(toMul);
            const auto vecA = _mm256_load_ps(
                static_cast<float const*>(&input[batchOffset + i]));
            const auto div = _mm256_mul_ps(vecA, vecMul);
            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i]), div);
        }
    }
}

void ScalarDivCpu(const Span<float> input, float toDiv, Span<float> out,
                  std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto vecMul = _mm256_set1_ps(toDiv);
            const auto vecA = _mm256_load_ps(
                static_cast<float const*>(&input[batchOffset + i]));
            const auto mul = _mm256_div_ps(vecA, vecMul);
            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i]), mul);
        }
    }
}

void SetCpu(Span<float> data, float toSet, std::size_t size,
            std::size_t batchSize)
{
    //#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; static_cast<std::size_t>(batchIdx) < batchSize;
         ++batchIdx)
    {
        const auto batchOffset = size * batchIdx;
        for (std::size_t i = 0; i < size; i += 8)
        {
            const auto set = _mm256_set1_ps(toSet);
            _mm256_store_ps(data.Address(batchOffset + i), set);
        }
    }
}
} // namespace Takion::Compute::CPU
