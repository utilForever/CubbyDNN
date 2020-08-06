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
//! \param numRow : number of rows of inputA, C and out
//! \param inputA: input A
//! \param inputB : input B
//! \param out: output
//! \param numCol : number of columns of inputB, C and out
//! \param numMiddle : number of columns of inputA and rows of inputB
//! \param batchSize : number of batches
template <typename T>
void MultiplyCpu(const Span<T> inputA, const Span<T> inputB, Span<T> out,
                 unsigned numRow,
                 unsigned numCol, unsigned numMiddle, unsigned batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void MultiplyWithBroadcastCpu(const Span<T> inputA, const Span<T> inputB,
                              Span<T> out,
                              unsigned numRow, unsigned numCol,
                              unsigned numMiddle,
                              unsigned batchSize, bool broadCastA)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void CpuTranspose(const Span<T> input, Span<T> output, unsigned numRow,
                  unsigned numCol, unsigned batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void ShrinkCpu(const Span<T> input, Span<T> output, unsigned size,
               unsigned batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void AddCpu(const Span<T> A, const Span<T> B, Span<T> out,
            unsigned size,
            unsigned batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

//! BroadCasts input B to A
template <typename T>
void AddWithBroadcastCpu(const Span<T> A, const Span<T> B,
                         Span<T> out, unsigned size, unsigned batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void DotCpu(const Span<T> inputA, const Span<T> inputB, Span<T> out,
            unsigned size,
            unsigned batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}

template <typename T>
void ScalarMulCpu(const Span<T> inputA, T toMul, Span<T> out, unsigned size,
                  unsigned batchSize)
{
    throw std::invalid_argument("Unsupported data type");
}
}

#endif
