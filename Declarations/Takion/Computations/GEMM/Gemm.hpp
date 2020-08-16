// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_GEMM_HPP
#define TAKION_COMPUTE_GEMM_HPP

#include <Takion/Utils/Span.hpp>

namespace Takion::Compute::CPU
{
using namespace Utils;

//! Performs out = AB
//! Data given to this function must be aligned by 256 bytes
//! \param numRowA : number of rows of inputA, C and out
//! \param inputA: input A
//! \param inputB : input B
//! \param out: output
//! \param numColA : number of columns of inputB, C and out
//! \param numRowB : number of columns of inputA and rows of inputB
//! \param numMatrices : number of batches
template <typename T>
void MultiplyCpu(const Span<T> inputA, const Span<T> inputB, Span<T> out,
                 std::size_t numRowA,
                 std::size_t numColA, std::size_t numRowB, std::size_t numColB,
                 std::size_t numMatrices)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void MultiplyWithBroadcastCpu(const Span<T> inputA,
                              const Span<T> inputB, Span<T> out,
                              std::size_t numRowA, std::size_t numColA,
                              std::size_t numRowB, std::size_t numColB,
                              std::size_t numMatrices, bool broadCastA)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void CpuTranspose(const Span<T> input, Span<T> output, std::size_t numRow,
                  std::size_t numCol, std::size_t batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void ShrinkCpu(const Span<T> input, Span<T> output, std::size_t size,
               std::size_t batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void AddCpu(const Span<T> A, const Span<T> B, Span<T> out,
            std::size_t size,
            std::size_t batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void SubCpu(const Span<T> A, const Span<T> B, Span<T> out, std::size_t size,
            std::size_t batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

//! BroadCasts input B to A
template <typename T>
void AddWithBroadcastCpu(const Span<T> A, const Span<T> B,
                         Span<T> out, std::size_t size, std::size_t batchSize,
                         bool broadCastA)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void SubWithBroadcastCpu(const Span<T> A, const Span<T> B, Span<T> out,
                         std::size_t size, std::size_t batchSize,
                         bool broadCastA)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void DotCpu(const Span<T> inputA, const Span<T> inputB, Span<T> out,
            std::size_t size,
            std::size_t batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void DotWithBroadcastCpu(const Span<T> inputA, const Span<T> inputB,
                         Span<T> out,
                         std::size_t size, std::size_t batchSize,
                         bool broadCastA)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void ScalarMulCpu(const Span<T> input, T toMul, Span<T> output,
                  std::size_t size,
                  std::size_t batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void ScalarDivCpu(const Span<T> input, T toDiv, Span<T> output,
                  std::size_t size,
                  std::size_t batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void SetCpu(Span<T> data, T toSet, std::size_t size, std::size_t batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T, typename Function>
void ApplyCpu(const Span<T> input, Span<T> output, Function function,
              std::size_t size, std::size_t batchSize)
{
#pragma omp parallel for schedule(static) default(shared)
    for (int batchIdx = 0; batchIdx < static_cast<int>(batchSize); batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 1)
        {
            output[batchOffset + i] = static_cast<T>(function(
                input[batchOffset + i]));
        }
    }
}
}

#endif
