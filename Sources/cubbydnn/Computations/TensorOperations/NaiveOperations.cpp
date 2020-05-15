// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Computations/TensorOperations/NaiveOperations.hpp>
#include <cubbydnn/Computations/TensorOperations/Naive.hpp>
#include <cubbydnn/computations/TensorOperations/Blaze.hpp>
#include <cubbydnn/Computations/Activations/ActivationFunc.hpp>

namespace CubbyDNN::Compute
{
void Native::Multiply(const Tensor& inputA, const Tensor& inputB,
                      Tensor& output)
{
    if (inputA.NumericType != inputB.NumericType ||
        inputA.NumericType != output.NumericType)
        throw std::runtime_error("Number system mismatches between tensors");

    if (inputA.Device.Type() != inputB.Device.Type() ||
        inputB.Device.Type() != output.Device.Type())
        throw std::runtime_error("Device Type mismatches between tensors");

    if (inputA.TensorShape.NumCols() != inputB.TensorShape.NumRows() ||
        inputA.TensorShape.NumRows() != output.TensorShape.NumRows() ||
        inputB.TensorShape.NumCols() != output.TensorShape.NumCols())
        throw std::runtime_error("Tensor shape mismatch");

    const auto numberSystem = inputA.NumericType;
    const auto deviceType = inputA.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Naive::TensorMul<float>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorMul<float>(inputA, inputB, output);
        else
            throw std::runtime_error("Not implemented");
    }
    else
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Naive::TensorMul<int>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorMul<int>(inputA, inputB, output);
        else
            throw std::runtime_error("Not implemented");
    }
}

void Native::Add(const Tensor& inputA, const Tensor& inputB,
                 Tensor& output)
{
    if (inputA.NumericType != inputB.NumericType ||
        inputA.NumericType != output.NumericType)
        throw std::runtime_error("Number system mismatches between tensors");

    if (inputA.Device.Type() != inputB.Device.Type() ||
        inputB.Device.Type() != output.Device.Type())
        throw std::runtime_error("Device Type mismatches between tensors");

    if (inputA.TensorShape != inputB.TensorShape ||
        inputB.TensorShape != output.TensorShape)
        throw std::runtime_error("Tensor shape mismatch");

    const auto numberSystem = inputA.NumericType;
    const auto deviceType = inputA.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Naive::TensorAdd<float>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorAdd<float>(inputA, inputB, output);
        else
            throw std::runtime_error("Not implemented");
    }
    else
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Naive::TensorAdd<int>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorAdd<int>(inputA, inputB, output);
        else
            throw std::runtime_error("Not implemented");
    }
}

void Native::Transpose(const Tensor& input, Tensor& output)
{
    if (input.NumericType != output.NumericType)
        throw std::runtime_error("Number system mismatch");
    auto transposedInputShape = input.TensorShape;
    transposedInputShape.Transpose();

    if (transposedInputShape != output.TensorShape)
        throw std::runtime_error("Tensor shape mismatch");

    const auto numberSystem = input.NumericType;
    if (numberSystem == NumberSystem::Float)
        Naive::TensorTranspose<float>(input, output);
    else
        Naive::TensorTranspose<int>(input, output);
}

void Native::Activation(const Tensor& input, Tensor& output,
                        ActivationFunc & activation)
{
}

void Native::Dot(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
}
} // namespace CubbyDNN
