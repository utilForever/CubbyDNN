// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_TEST_COMPUTETEST_HPP
#define TAKION_TEST_COMPUTETEST_HPP

#include <Takion/Computations/GEMM/MathKernel.hpp>
//TODO : Move Device.hpp to Utils folder
#include <Takion/Computations/Device.hpp>
#include <Takion/Computations/Initializers/InitializerType.hpp>
#include "SolidComputations.hpp"
#include <doctest.h>

namespace Takion::Test
{
template <typename T>
void TestMultiply(Compute::Device device)
{
    const auto batchSize = 100;
    const auto numRow = 120;
    const auto numCol = 130;
    const auto numMiddle = 150;

    Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                     static_cast<T>(10));
    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numMiddle });
    Shape shapeB({ numMiddle, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, batchSize, device);
    Tensor<T> B(shapeB, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);
    Tensor<T> truth(shapeOut, batchSize, device);

    randomNormalInitializer.Initialize(A);
    randomNormalInitializer.Initialize(B);
    zeroInitializer.Initialize(result);
    zeroInitializer.Initialize(truth);

    Compute::Multiply(A, B, result);
    Test::Multiply(A, B, truth);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
            {
                CHECK(result.At(batchIdx, { rowIdx, colIdx }) ==
                    truth.At(batchIdx, { rowIdx, colIdx }));
            }
}

template <typename T>
void TestTranspose(Compute::Device device)
{
    const auto batchSize = 100;
    const auto numRow = 120;
    const auto numCol = 130;

    Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                     static_cast<T>(10));
    Compute::Zeros<T> zeroInitializer;

    Shape shapeIn({ 10, numRow, numCol });
    Shape shapeOut({ 10, numCol, numRow });

    Tensor<T> in(shapeIn, batchSize, device);
    Tensor<T> truth(shapeOut, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);

    randomNormalInitializer.Initialize(in);
    zeroInitializer.Initialize(result);
    zeroInitializer.Initialize(truth);

    Compute::Transpose(in, result);
    Test::Transpose(in, truth);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
            {
                CHECK(result.At(batchIdx, { colIdx, rowIdx }) ==
                    truth.At(batchIdx, { colIdx, rowIdx }));
            }
}

template <typename T>
void TestShrink(Compute::Device device)
{
    const auto batchSize = 100;
    const auto numRow = 120;
    const auto numCol = 130;

    Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                     static_cast<T>(10));
    Compute::Zeros<T> zeroInitializer;

    Shape shapeIn({ 10, numRow, numCol });
    Shape shapeOut({ 10, numRow, numCol });

    Tensor<T> in(shapeIn, batchSize, device);
    Tensor<T> truth(shapeOut, device);
    Tensor<T> result(shapeOut, device);

    randomNormalInitializer.Initialize(in);
    zeroInitializer.Initialize(result);
    zeroInitializer.Initialize(truth);

    Compute::Shrink(in, result);
    Test::Shrink(in, truth);

    for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
        for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
        {
            CHECK(result.At(0, { rowIdx, colIdx }) ==
                truth.At(0, { rowIdx, colIdx }));
        }
}

template <typename T>
void TestAdd(Compute::Device device)
{
    const auto batchSize = 100;
    const auto numRow = 120;
    const auto numCol = 130;

    Compute::RandomNormal<T> randomNormalInitializer(static_cast<T>(-10),
                                                     static_cast<T>(10));
    Compute::Zeros<T> zeroInitializer;

    Shape shapeA({ numRow, numCol });
    Shape shapeB({ numRow, numCol });
    Shape shapeOut({ numRow, numCol });

    Tensor<T> A(shapeA, batchSize, device);
    Tensor<T> B(shapeB, batchSize, device);

    Tensor<T> truth(shapeOut, batchSize, device);
    Tensor<T> result(shapeOut, batchSize, device);

    randomNormalInitializer.Initialize(A);
    randomNormalInitializer.Initialize(B);

    zeroInitializer.Initialize(A);
    zeroInitializer.Initialize(B);

    Compute::Add(A, B, result);
    Test::Add(A, B, truth);

    for (std::size_t idx = 0; idx < shapeOut.Size() * batchSize; ++idx)
        CHECK(result.At(idx) == truth.At(idx));
}
}

#endif
