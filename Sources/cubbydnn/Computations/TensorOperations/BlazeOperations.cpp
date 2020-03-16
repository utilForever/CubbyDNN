// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/computations/TensorOperations/BlazeOperations.hpp>
#include <cubbydnn/computations/TensorOperations/Blaze.hpp>

namespace CubbyDNN
{
BlazeOperation::BlazeOperation()
    : TensorOperation()
{
}

void BlazeOperation::Multiply(const Tensor& inputA, const Tensor& inputB,
                              Tensor& output)
{
    if (inputA.Info.GetNumberSystem() != inputB.Info.GetNumberSystem() ||
        inputA.Info.GetNumberSystem() != output.Info.GetNumberSystem())
        throw std::runtime_error("Number system mismatches between tensors");

    const auto numberSystem = inputA.Info.GetNumberSystem();

    if (numberSystem == NumberSystem::Float)
        Blaze::TensorMul<float>(inputA, inputB, output);
    else
        Blaze::TensorMul<int>(inputA, inputB, output);
}

void BlazeOperation::Add(const Tensor& inputA, const Tensor& inputB,
                         Tensor& output)
{
    if (inputA.Info.GetNumberSystem() != inputB.Info.GetNumberSystem() ||
        inputA.Info.GetNumberSystem() != output.Info.GetNumberSystem())
        throw std::runtime_error("Number system mismatches between tensors");

    const auto numberSystem = inputA.Info.GetNumberSystem();
    if (numberSystem == NumberSystem::Float)
        Blaze::TensorAdd<float>(inputA, inputB, output);
    else
        Blaze::TensorAdd<int>(inputA, inputB, output);
}

void BlazeOperation::Transpose(const Tensor& input, Tensor& output)
{
    if (input.Info.GetNumberSystem() != output.Info.GetNumberSystem())
        throw std::runtime_error("Number system mismatch");
    const auto numberSystem = input.Info.GetNumberSystem();

    if (numberSystem == NumberSystem::Float)
        Blaze::TensorTranspose<float>(input, output);
    else
        Blaze::TensorTranspose<int>(input, output);
}
} // namespace CubbyDNN
