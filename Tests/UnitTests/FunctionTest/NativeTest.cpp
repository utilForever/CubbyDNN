/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "NativeTest.hpp"
#include "gtest/gtest.h"

#include <cubbydnn/Computations/TensorOperations/Computations.hpp>
#include <array>

namespace CubbyDNN::Test
{
void TestMatMul()
{
    Compute::Device device(0, Compute::DeviceType::Cpu, "testDevice", 0);
    const auto batchSize = 3;

    Tensor tensorA({ batchSize, 3, 2 }, device);
    Tensor tensorB({ batchSize, 2, 3 }, device);

    Tensor output({ batchSize, 3, 3 }, device);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        tensorA.At<float>({ batchIdx, 0, 0 }) = 1.0f;
        tensorA.At<float>({ batchIdx, 0, 1 }) = 2.0f;
        tensorA.At<float>({ batchIdx, 1, 0 }) = 3.0f;
        tensorA.At<float>({ batchIdx, 1, 1 }) = 4.0f;
        tensorA.At<float>({ batchIdx, 2, 0 }) = 5.0f;
        tensorA.At<float>({ batchIdx, 2, 1 }) = 6.0f;

        tensorB.At<float>({ batchIdx, 0, 0 }) = 1.0f;
        tensorB.At<float>({ batchIdx, 0, 1 }) = 3.0f;
        tensorB.At<float>({ batchIdx, 0, 2 }) = 5.0f;
        tensorB.At<float>({ batchIdx, 1, 0 }) = 2.0f;
        tensorB.At<float>({ batchIdx, 1, 1 }) = 4.0f;
        tensorB.At<float>({ batchIdx, 1, 2 }) = 6.0f;
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
                auto num = output.At<float>({ batchIdx, i, j });

                EXPECT_EQ(ans, num);
            }
        }
}

void TestMatMul2()
{
    Compute::Device device(0, Compute::DeviceType::Cpu, "testDevice", 0);

    const auto batchSize = 3;
    const auto size = 150;

    Tensor tensorA({ batchSize, size, size },
                   device);
    Tensor tensorB({ batchSize, size, size },
                   device);

    Tensor output({ batchSize, size, size }, device);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t i = 0; i < size; i++)
        {
            for (std::size_t j = 0; j < size; j++)
            {
                tensorA.At<float>({ batchIdx, i, j }) = 2.0f;
                tensorB.At<float>({ batchIdx, i, j }) = 2.0f;
            }
        }

    Compute::Multiply(tensorA, tensorB, output);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t i = 0; i < size; i++)
        {
            for (std::size_t j = 0; j < size; j++)
            {
                const auto num = output.At<float>({ batchIdx, i, j });
                EXPECT_EQ(num, 4 * size);
            }
        }
}

void TestMatAdd()
{
    Compute::Device device(0, Compute::DeviceType::Cpu, "testDevice", 0);
    const auto batchSize = 3;
    const auto rowSize = 2;
    const auto colSize = 5;

    Tensor tensorA(
        { batchSize, rowSize, colSize }, device);
    Tensor tensorB(
        { batchSize, rowSize, colSize }, device);

    Tensor output({ batchSize, rowSize, colSize }, device);

    for (std::size_t batch = 0; batch < batchSize; ++batch)
        for (std::size_t i = 0; i < rowSize; i++)
        {
            for (std::size_t j = 0; j < colSize; j++)
            {
                tensorA.At<float>({ batch, i, j }) = 4.0f;
                tensorB.At<float>({ batch, i, j }) = 4.0f;
            }
        }

    Compute::Add(tensorA, tensorB, output);

    for (std::size_t batch = 0; batch < batchSize; ++batch)
        for (std::size_t i = 0; i < rowSize; i++)
        {
            for (std::size_t j = 0; j < colSize; j++)
            {
                const auto num = output.At<float>({ batch, i, j });
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

    Tensor tensorA({ batchSize, rowSize, colSize }, device);
    Tensor tensorB({ batchSize, rowSize, colSize }, device);

    Tensor output({ batchSize, rowSize, colSize }, device);

    for (std::size_t batch = 0; batch < batchSize; ++batch)
        for (std::size_t i = 0; i < rowSize; i++)
        {
            for (std::size_t j = 0; j < colSize; j++)
            {
                tensorA.At<float>({ batch, i, j }) = 4.0f;
                tensorB.At<float>({ batch, i, j }) = 4.0f;
            }
        }

    Compute::Dot(tensorA, tensorB, output);

    for (std::size_t batch = 0; batch < batchSize; ++batch)
        for (std::size_t i = 0; i < rowSize; i++)
        {
            for (std::size_t j = 0; j < colSize; j++)
            {
                const auto num = output.At<float>({ batch, i, j });
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
