// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/computations/Functions/Matrix.hpp>
#include <cubbydnn/computations/Functions/MatrixOps.hpp>

namespace CubbyDNN
{
void MultiplyOp(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    assert(inputA.Info.GetNumberSystem() == inputB.Info.GetNumberSystem() &&
           inputA.Info.GetNumberSystem() == output.Info.GetNumberSystem());

    const auto numberSystem = inputA.Info.GetNumberSystem();
    switch (numberSystem)
    {
        case NumberSystem::Float64:
            TensorMul<double>(inputA, inputB, output);
            break;
        case NumberSystem::Float32:
            TensorMul<float>(inputA, inputB, output);
            break;
        case NumberSystem::Int64:
            TensorMul<std::int64_t>(inputA, inputB, output);
            break;
        case NumberSystem::Int32:
            TensorMul<std::int32_t>(inputA, inputB, output);
            break;
        case NumberSystem::Int16:
            TensorMul<std::int16_t>(inputA, inputB, output);
            break;
        case NumberSystem::Int8:
            TensorMul<std::int8_t>(inputA, inputB, output);
            break;
        default:
            throw std::runtime_error("UnsupportedNumberSystemError");
    }
}

void AddOp(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    assert(inputA.Info.GetNumberSystem() == inputB.Info.GetNumberSystem() &&
           inputA.Info.GetNumberSystem() == output.Info.GetNumberSystem());

    const auto numberSystem = inputA.Info.GetNumberSystem();
    switch (numberSystem)
    {
        case NumberSystem::Float64:
            TensorAdd<double>(inputA, inputB, output);
            break;
        case NumberSystem::Float32:
            TensorAdd<float>(inputA, inputB, output);
            break;
        case NumberSystem::Int64:
            TensorAdd<std::int64_t>(inputA, inputB, output);
            break;
        case NumberSystem::Int32:
            TensorAdd<std::int32_t>(inputA, inputB, output);
            break;
        case NumberSystem::Int16:
            TensorAdd<std::int16_t>(inputA, inputB, output);
            break;
        case NumberSystem::Int8:
            TensorAdd<std::int8_t>(inputA, inputB, output);
            break;
        default:
            throw std::runtime_error("UnsupportedNumberSystemError");
    }
}

void TransposeOp(const Tensor& input, Tensor& output)
{
    assert(input.Info.GetNumberSystem() == output.Info.GetNumberSystem());
    const auto numberSystem = input.Info.GetNumberSystem();
    switch (numberSystem)
    {
        case NumberSystem::Float64:
            TensorTranspose<double>(input, output);
            break;
        case NumberSystem::Float32:
            TensorTranspose<float>(input, output);
            break;
        case NumberSystem::Int64:
            TensorTranspose<long>(input, output);
            break;
        case NumberSystem::Int32:
            TensorTranspose<int>(input, output);
            break;

        default:
            assert(false && "UnsupportedNumberSystemError");
    }
}

void ReshapeOp(const Tensor& input, Tensor& output)
{
    assert(input.Info.GetNumberSystem() == output.Info.GetNumberSystem());
    assert(input.Info.GetByteSize() == output.Info.GetByteSize());
    std::memcpy(output.DataPtr, input.DataPtr, input.Info.GetByteSize());
}
}  // namespace CubbyDNN
