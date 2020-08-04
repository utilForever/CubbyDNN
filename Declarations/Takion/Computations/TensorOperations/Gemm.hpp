// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_GEMM_HPP
#define TAKION_COMPUTE_GEMM_HPP

#include <Takion/Utils/Span.hpp>
#include <immintrin.h>
#include <algorithm>

namespace Takion::Compute
{
using namespace Utils;

//! Performs out = AB
//! Data given to this function must be aligned by 256 bytes
//! \param numRow : number of rows of inputA, C and out
//! \param numCol : number of columns of inputB, C and out
//! \param numMiddle : number of columns of inputA and rows of inputB
//! \param batchSize : number of batches
template <typename T>
void Multiply(const Span<T> inputA, const Span<T> inputB, Span<T> out,
              unsigned numRow,
              unsigned numCol, unsigned numMiddle, unsigned batchSize)
{
    throw std::runtime_error("Unsupported data type");
}

template <>
inline void Multiply(const Span<int> inputA, const Span<int> inputB,
                     Span<int> out,
                     unsigned numRow, unsigned numCol, unsigned numMiddle,
                     unsigned batchSize)
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
                                (__m256i*)&inputB[batchOffsetB + k * numRow + j
                                ]);
                            auto vecB_mat2 = _mm256_loadu_si256((
                                __m256i*)&inputB[
                                batchOffsetB + k * numRow + j + 8]);
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
inline void Multiply(const Span<float> inputA, const Span<float> inputB,
                     Span<float> out,
                     unsigned numRow, unsigned numCol, unsigned numMiddle,
                     unsigned batchSize)
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
                                    &inputB[batchOffsetB + k * numRow + j + 8]
                                ));
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

template <typename T>
void Add(const Span<T> inputA, const Span<T> inputB, Span<T> out, unsigned size,
         unsigned batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <>
void Add(const Span<int> A, const Span<int> B, Span<int> Dest, unsigned size,
         unsigned batchSize)
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

            _mm256_storeu_si256((__m256i*)&Dest[batchOffset + i], sum1);
            _mm256_storeu_si256((__m256i*)&Dest[batchOffset + i + 8], sum2);
        }
    }
}

template <>
void Add(const Span<float> inputA, const Span<float> inputB, Span<float> out,
         unsigned size, unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecA1 = _mm256_load_ps(
                static_cast<float const*>(&inputA[batchOffset + i]));
            const auto vecA2 = _mm256_load_ps(static_cast<float const*>(
                &inputA[batchOffset + i + 8]));

            const auto vecB1 = _mm256_load_ps(
                static_cast<float const*>(&inputB[batchOffset + i]));
            const auto vecB2 = _mm256_load_ps(static_cast<float const*>(
                &inputB[batchOffset + i + 8]));

            const auto sum1 = _mm256_add_ps(vecA1, vecB1);
            const auto sum2 = _mm256_add_ps(vecA2, vecB2);

            _mm256_store_ps(
                static_cast<float*>(&out[batchOffset + i]), sum1);
            _mm256_store_ps(
                static_cast<float*>(&out[batchOffset + i + 8]),
                sum2);
        }
    }
}

template <typename T>
void Dot(const Span<T> inputA, const Span<T> inputB, Span<T> out, unsigned size,
         unsigned batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <>
void Dot(const Span<int> inputA, const Span<int> inputB, Span<int> out,
         unsigned size,
         unsigned batchSize)
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
void Dot(const Span<float> inputA, const Span<float> inputB, Span<float> out,
         unsigned size, unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecA1 =
                _mm256_load_ps(
                    static_cast<float const*>(&inputA[batchOffset + i]));
            const auto vecA2 = _mm256_load_ps(
                static_cast<float const*>(&inputA[batchOffset + i + 8]));

            const auto vecB1 =
                _mm256_load_ps(
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

template <typename T>
void ScalarMul(const Span<T> inputA, T toMul, Span<T> out, unsigned size,
               unsigned batchSize)
{
    throw std::runtime_error("Unsupported data type");
}

template <>
inline void ScalarMul(const Span<int> inputA, int toMul, Span<int> out,
                      unsigned size,
                      unsigned batchSize)
{
#pragma parallel for schedule(static) default(shared)
    for (unsigned batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 16)
        {
            const auto vecMul =
                _mm256_set1_epi32(toMul);
            const auto vecA1 =
                _mm256_loadu_si256((__m256i*)&inputA[batchOffset + i]);
            const auto vecA2 =
                _mm256_loadu_si256((__m256i*)&inputA[batchOffset + i + 8]);

            const auto mul1 = _mm256_mullo_epi32(vecA1, vecMul);
            const auto mul2 = _mm256_mullo_epi32(vecA2, vecMul);

            _mm256_storeu_si256((__m256i*)&out[batchOffset + i], mul1);
            _mm256_storeu_si256((__m256i*)&out[batchOffset + i + 8], mul2);
        }
    }
}

template <>
inline void ScalarMul(const Span<float> inputA, float toMul, Span<float> out,
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
                static_cast<float const*>(&inputA[batchOffset + i]));
            const auto vecA2 = _mm256_load_ps(
                static_cast<float const*>(&inputA[batchOffset + i + 8]));

            const auto mul1 = _mm256_mul_ps(vecA1, vecMul);
            const auto mul2 = _mm256_mul_ps(vecA2, vecMul);

            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i]), mul1);
            _mm256_store_ps(static_cast<float*>(&out[batchOffset + i + 8]),
                            mul2);
        }
    }
}
}

#endif
