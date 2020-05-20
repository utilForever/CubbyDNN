/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "MatrixTest.hpp"
#include <cubbydnn/Computations/TensorOperations/Computations.hpp>
#include <iostream>
#include <array>
#include "gtest/gtest.h"

namespace CubbyDNN
{
void TestMatMul()
{
    Compute::Device device(0, Compute::DeviceType::Cpu, "testDevice", 0);
    const auto batchSize = 3;

    Tensor tensorA({ 2, 3, batchSize }, device);
    Tensor tensorB({ 3, 2, batchSize }, device);

    Tensor output({ 3, 3, batchSize }, device);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        tensorA.At<float>({ 0, 0, batchIdx }) = 1.0f;
        tensorA.At<float>({ 1, 0, batchIdx }) = 2.0f;
        tensorA.At<float>({ 0, 1, batchIdx }) = 3.0f;
        tensorA.At<float>({ 1, 1, batchIdx }) = 4.0f;
        tensorA.At<float>({ 0, 2, batchIdx }) = 5.0f;
        tensorA.At<float>({ 1, 2, batchIdx }) = 6.0f;

        tensorB.At<float>({ 0, 0, batchIdx }) = 1.0f;
        tensorB.At<float>({ 1, 0, batchIdx }) = 3.0f;
        tensorB.At<float>({ 2, 0, batchIdx }) = 5.0f;
        tensorB.At<float>({ 0, 1, batchIdx }) = 2.0f;
        tensorB.At<float>({ 1, 1, batchIdx }) = 4.0f;
        tensorB.At<float>({ 2, 1, batchIdx }) = 6.0f;
    }

    Compute::Multiply(tensorA, tensorB, output);

    std::array<std::array<float, 3>, 3> answer = {
        { { 5.0f, 11.0f, 17.0f },
          { 11.0f, 25.0f, 39.0f },
          { 17.0f, 39.0f, 61.0f } }
    };

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t i = 0; i < 3; i++)
        {
            for (std::size_t j = 0; j < 3; j++)
            {
                auto ans = answer[i][j];
                auto num = output.At<float>({ j, i, batchIdx });

                EXPECT_EQ(ans, num);
            }
        }
}

void TestMatMul2()
{
    Compute::Device device(0, Compute::DeviceType::Cpu, "testDevice", 0);

    const auto batchSize = 3;
    const auto size = 3;

    Tensor tensorA({ size, size, batchSize },
                   device);
    Tensor tensorB({ size, size, batchSize },
                   device);

    Tensor output({ size, size, batchSize }, device);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t i = 0; i < size; i++)
        {
            for (std::size_t j = 0; j < size; j++)
            {
                tensorA.At<float>({ j, i, batchIdx }) = 2.0f;
                tensorB.At<float>({ j, i, batchIdx }) = 2.0f;
            }
        }

    Compute::Multiply(tensorA, tensorB, output);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t i = 0; i < size; i++)
        {
            for (std::size_t j = 0; j < size; j++)
            {
                const auto num = output.At<float>({ j, i, batchIdx });
                EXPECT_EQ(num, 12);
            }
        }
}

void TestMatAdd()
{
    Compute::Device device(0, Compute::DeviceType::Cpu, "testDevice", 0);
    const auto batchSize = 3;
    const auto rowSize = 2;
    const auto colSize = 5;

    Tensor tensorA({ rowSize, colSize, batchSize }, device);
    Tensor tensorB({ rowSize, colSize, batchSize }, device);

    Tensor output({ rowSize, colSize, batchSize }, device);

    for (std::size_t batch = 0; batch < batchSize; ++batch)
        for (std::size_t i = 0; i < rowSize; i++)
        {
            for (std::size_t j = 0; j < colSize; j++)
            {
                tensorA.At<float>({ i, j, batch }) = 4.0f;
                tensorB.At<float>({ i, j, batch }) = 4.0f;
            }
        }

    Compute::Add(tensorA, tensorB, output);

    for (std::size_t batch = 0; batch < batchSize; ++batch)
        for (std::size_t i = 0; i < rowSize; i++)
        {
            for (std::size_t j = 0; j < colSize; j++)
            {
                const auto num = output.At<float>({ i, j, batch });
                EXPECT_EQ(num, 8);
            }
        }
}

void TestMatDot()
{
    Compute::Device device(0, Compute::DeviceType::Cpu, "testDevice", 0);
    const auto batchSize = 3;
    const auto rowSize = 2;
    const auto colSize = 5;

    Tensor tensorA({ rowSize, colSize, batchSize }, device);
    Tensor tensorB({ rowSize, colSize, batchSize }, device);

    Tensor output({ rowSize, colSize, batchSize }, device);

    for (std::size_t batch = 0; batch < batchSize; ++batch)
        for (std::size_t i = 0; i < rowSize; i++)
        {
            for (std::size_t j = 0; j < colSize; j++)
            {
                tensorA.At<float>({ i, j, batch }) = 4.0f;
                tensorB.At<float>({ i, j, batch }) = 4.0f;
            }
        }

    Compute::Dot(tensorA, tensorB, output);

    for (std::size_t batch = 0; batch < batchSize; ++batch)
        for (std::size_t i = 0; i < rowSize; i++)
        {
            for (std::size_t j = 0; j < colSize; j++)
            {
                const auto num = output.At<float>({ i, j, batch });
                EXPECT_EQ(num, 16);
            }
        }
}

TEST(NativeOpTest, MatMul1)
{
    TestMatMul();
}

TEST(NativeOpTest, MatMul2)
{
    TestMatMul2();
}


TEST(NativeOpTest, MatAdd)
{
    TestMatAdd();
}


TEST(NativeOpTest, MatDot)
{
    TestMatDot();
}
} // namespace CubbyDNN
