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

//! Performs Dest = AB
//! Data given to this function must be aligned by 256 bytes
//! \param numRow : number of rows of A, C and Dest
//! \param numCol : number of columns of B, C and Dest
//! \param numMiddle : number of columns of A and rows of B
//! \param batchSize : number of batches
template <typename T>
void Multiply(const Span<T> A, const Span<T> B,
              Span<T> Dest, unsigned numRow, unsigned numCol,
              unsigned numMiddle, unsigned batchSize)
{
    throw std::runtime_error("Unsupported type");
}


template <>
inline void Multiply(const Span<int> A, const Span<int> B,
                     Span<int> Dest, unsigned numRow,
                     unsigned numCol, unsigned numMiddle,
                     unsigned batchSize)
{
    const auto jb = std::min(512u, numCol);
    const auto kb = std::min(24u, numRow);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
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
                                (__m256i*)(&Dest[i * numCol + j]));
                            sumB_1 = _mm256_load_si256(
                                (__m256i*)&Dest[i * numCol + j + 8]);
                        }
                        const auto limit = std::min(
                            static_cast<std::size_t>(numMiddle),
                            kk + kb);
                        for (size_t k = kk; k < limit; k++)
                        {
                            auto bc_mat1_1 =
                                _mm256_set1_epi32(A[i * numMiddle + k]);
                            auto vecA_mat2 =
                                _mm256_loadu_si256(
                                    (__m256i*)&B[k * numRow + j]);
                            auto vecB_mat2 =
                                _mm256_loadu_si256(
                                    (__m256i*)&B[k * numRow + j + 8]);
                            sumA_1 = _mm256_add_epi32(
                                sumA_1,
                                _mm256_mullo_epi32(bc_mat1_1, vecA_mat2));
                            sumB_1 = _mm256_add_epi32(
                                sumB_1,
                                _mm256_mullo_epi32(bc_mat1_1, vecB_mat2));
                        }
                        _mm256_storeu_si256((__m256i*)&Dest[i * numCol + j],
                                            sumA_1);
                        _mm256_storeu_si256((__m256i*)&Dest[i * numCol + j + 8],
                                            sumB_1);
                    }
                }
            }
        }
}

template <>
inline void Multiply(const Span<float> A, const Span<float> B,
                     Span<float> Dest, unsigned numRow, unsigned numCol,
                     unsigned numMiddle, unsigned batchSize)
{
    const auto jb = std::min(512u, numCol);
    const auto kb = std::min(24u, numRow);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
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
                        {
                            sumA_1 = sumB_1 = _mm256_setzero_ps();
                        }
                        else
                        {
                            sumA_1 = _mm256_load_ps(
                                static_cast<float const*>(&Dest[i * numCol + j]
                                ));
                            sumB_1 = _mm256_load_ps(
                                static_cast<float const*>(&Dest[
                                    i * numCol + j + 8]));
                        }
                        const auto limit = std::min(
                            static_cast<std::size_t>(numMiddle), kk + kb);
                        for (size_t k = kk; k < limit; k++)
                        {
                            const auto bc_mat1_1 =
                                _mm256_set1_ps(A[i * numMiddle + k]);
                            const auto vecA_mat2 = _mm256_load_ps(
                                static_cast<float const*>(&B[k * numRow + j]));
                            const auto vecB_mat2 = _mm256_load_ps(
                                static_cast<float const*>(&B[k * numRow + j + 8]
                                ));
                            sumA_1 = _mm256_add_ps(
                                sumA_1,
                                _mm256_mul_ps(bc_mat1_1, vecA_mat2));
                            sumB_1 = _mm256_add_ps(
                                sumB_1,
                                _mm256_mul_ps(bc_mat1_1, vecB_mat2));
                        }
                        _mm256_store_ps(
                            static_cast<float*>(&Dest[i * numCol + j]),
                            sumA_1);
                        _mm256_store_ps(
                            static_cast<float*>(&Dest[i * numCol + j + 8]),
                            sumB_1);
                    }
                }
            }
        }
}

}

#endif
