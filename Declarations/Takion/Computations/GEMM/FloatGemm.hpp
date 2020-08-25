// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_FLOATGEMM_HPP
#define TAKION_COMPUTE_FLOATGEMM_HPP

#include <Takion/Utils/Span.hpp>

namespace Takion::Compute::CPU::Float
{
using namespace Util;
void MultiplyCpu(const Span<float> inputA, const Span<float> inputB,
                 Span<float> out, std::size_t numRowA, std::size_t numColA,
                 std::size_t numRowB, std::size_t numColB,
                 std::size_t numMatrices);

void MultiplyWithBroadcastCpu(const Span<float> inputA,
                              const Span<float> inputB, Span<float> out,
                              std::size_t numRowA, std::size_t numColA,
                              std::size_t numRowB, std::size_t numColB,
                              std::size_t numMatrices, bool broadCastA);

void ShrinkCpu(const Span<float> input, Span<float> output, std::size_t size,
               std::size_t batchSize);

void AddCpu(const Span<float> inputA, const Span<float> inputB, Span<float> out,
            std::size_t size, std::size_t batchSize);

void SubCpu(const Span<float> A, const Span<float> B, Span<float> out,
            std::size_t size, std::size_t batchSize);

void AddWithBroadcastCpu(const Span<float> A, const Span<float> B,
                         Span<float> out, std::size_t size,
                         std::size_t batchSize, bool broadCastA);

void SubWithBroadcastCpu(const Span<float> A, const Span<float> B,
                         Span<float> out, std::size_t size,
                         std::size_t batchSize, bool broadCastA);

void DotCpu(const Span<float> inputA, const Span<float> inputB, Span<float> out,
            std::size_t size, std::size_t batchSize);

void DotWithBroadcastCpu(const Span<float> inputA, const Span<float> inputB,
                         Span<float> out, std::size_t size,
                         std::size_t batchSize, bool broadCastA);

void DivCpu(const Span<float> inputA, const Span<float> inputB, Span<float> out,
            std::size_t size, std::size_t batchSize);

void DivWithBroadcastCpu(const Span<float> inputA, const Span<float> inputB,
                         Span<float> out, std::size_t size,
                         std::size_t batchSize, bool broadCastA);

void ScalarMulCpu(const Span<float> input, float toMul, Span<float> out,
                  std::size_t size, std::size_t batchSize);

void ScalarDivCpu(const Span<float> input, float toDiv, Span<float> out,
                         std::size_t size, std::size_t batchSize);

void SetCpu(Span<float> data, float toSet, std::size_t size,
            std::size_t batchSize);
}

#endif
