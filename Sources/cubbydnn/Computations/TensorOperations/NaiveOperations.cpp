// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Computations/TensorOperations/NaiveOperations.hpp>
#include <cubbydnn/Computations/TensorOperations/Naive.hpp>

namespace CubbyDNN
{
void Native::Multiply(const Tensor& inputA, const Tensor& inputB,
                              Tensor& output)
{
    if (inputA.NumericType != inputB.NumericType ||
        inputA.NumericType != output.NumericType)
        throw std::runtime_error("Number system mismatches between tensors");

    const auto numberSystem = inputA.NumericType;

    if (numberSystem == NumberSystem::Float)
        Naive::TensorMul<float>(inputA, inputB, output);
    else
        Naive::TensorMul<int>(inputA, inputB, output);
}

void Native::Add(const Tensor& inputA, const Tensor& inputB,
                         Tensor& output)
{
    if (inputA.NumericType != inputB.NumericType ||
        inputA.NumericType != output.NumericType)
        throw std::runtime_error("Number system mismatches between tensors");

    const auto numberSystem = inputA.NumericType;

    if (numberSystem == NumberSystem::Float)
        Naive::TensorAdd<float>(inputA, inputB, output);
    else
        Naive::TensorAdd<int>(inputA, inputB, output);
}

void Native::Transpose(const Tensor& input, Tensor& output)
{
    if (input.NumericType != output.NumericType)
        throw std::runtime_error("Number system mismatch");

    const auto numberSystem = input.NumericType;
    if (numberSystem == NumberSystem::Float)
        Naive::TensorTranspose<float>(input, output);
    else
        Naive::TensorTranspose<int>(input, output);
}
} // namespace CubbyDNN
