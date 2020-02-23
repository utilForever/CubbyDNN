/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "MatrixTest.hpp"
#include "gtest/gtest.h"
#include <cubbydnn/Computations/Functions/Matrix.hpp>
#include <iostream>

namespace CubbyDNN
{
void TestMatMul()
{
    Tensor tensorA =
        AllocateTensor(TensorInfo({ 1, 1, 3, 3 }, NumberSystem::Float32));
    Tensor tensorB =
        AllocateTensor(TensorInfo({ 1, 1, 3, 3 }, NumberSystem::Float32));

    SetData<float>({ 0, 0, 0, 0 }, tensorA, 4.0f);
    SetData<float>({ 0, 0, 1, 1 }, tensorA, 4.0f);
    SetData<float>({ 0, 0, 2, 2 }, tensorA, 4.0f);

    SetData<float>({ 0, 0, 0, 0 }, tensorB, 4.0f);
    SetData<float>({ 0, 0, 1, 1 }, tensorB, 4.0f);
    SetData<float>({ 0, 0, 2, 2 }, tensorB, 4.0f);

    Tensor output =
        AllocateTensor(TensorInfo({ 1, 1, 3, 3 }, NumberSystem::Float32));

    Multiply(tensorA, tensorB, output);

    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            const auto num = GetData<float>({ 0, 0, i, j }, output);
            std::cout << num << " ";
        }
        std::cout << std::endl;
    }
}

TEST(MatrixTest, MatMul)
{
    TestMatMul();
}
} // namespace CubbyDNN
