// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Computations/TensorOperations/NaiveOperations.hpp>
#include <cubbydnn/Computations/TensorOperations/Naive.hpp>

namespace CubbyDNN
{
NaiveOperation::NaiveOperation()
    : TensorOperation()
{
}


void NaiveOperation::Multiply(const Tensor& inputA, const Tensor& inputB,
                              Tensor& output)
{
    if (inputA.Info.GetNumberSystem() != inputB.Info.GetNumberSystem() ||
        inputA.Info.GetNumberSystem() != output.Info.GetNumberSystem())
        throw std::runtime_error("Number system mismatches between tensors");

    const auto numberSystem = inputA.Info.GetNumberSystem();

    if (numberSystem == NumberSystem::Float)
        Naive::TensorMul<float>(inputA, inputB, output);
    else
        Naive::TensorMul<int>(inputA, inputB, output);
}

void NaiveOperation::Add(const Tensor& inputA, const Tensor& inputB,
                         Tensor& output)
{
    if (inputA.Info.GetNumberSystem() != inputB.Info.GetNumberSystem() ||
        inputA.Info.GetNumberSystem() != output.Info.GetNumberSystem())
        throw std::runtime_error("Number system mismatches between tensors");

    const auto numberSystem = inputA.Info.GetNumberSystem();

    if (numberSystem == NumberSystem::Float)
        Naive::TensorAdd<float>(inputA, inputB, output);
    else
        Naive::TensorAdd<int>(inputA, inputB, output);
}

void NaiveOperation::Transpose(const Tensor& input, Tensor& output)
{
    if (input.Info.GetNumberSystem() != output.Info.GetNumberSystem())
        throw std::runtime_error("Number system mismatch");

    const auto numberSystem = input.Info.GetNumberSystem();
    if (numberSystem == NumberSystem::Float)
        Naive::TensorTranspose<float>(input, output);
    else
        Naive::TensorTranspose<int>(input, output);
}
} // namespace CubbyDNN
