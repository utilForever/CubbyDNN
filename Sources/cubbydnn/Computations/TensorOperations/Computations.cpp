// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <iostream>
#include <cubbydnn/Computations/TensorOperations/Computations.hpp>
#include <cubbydnn/Computations/TensorOperations/Native.hpp>
#include <cubbydnn/Computations/TensorOperations/Blaze.hpp>

namespace CubbyDNN::Compute
{
void Multiply(const Tensor& inputA, const Tensor& inputB,
              Tensor& output)
{
    if (inputA.NumericType != inputB.NumericType ||
        inputA.NumericType != output.NumericType)
        throw std::runtime_error(
            "Multiply - Number system mismatches between tensors");

    if (inputA.Device != inputB.Device ||
        inputB.Device != output.Device)
        throw std::runtime_error(
            "Multiply - Device Type mismatches between tensors");

    if (inputA.TensorShape.NumCols() != inputB.TensorShape.NumRows() ||
        inputA.TensorShape.NumRows() != output.TensorShape.NumRows() ||
        inputB.TensorShape.NumCols() != output.TensorShape.NumCols())
        throw std::runtime_error("Multiply - Tensor shape mismatch");

    if (inputA.TensorShape.BatchSize() != inputB.TensorShape.BatchSize() ||
        inputB.TensorShape.BatchSize() != output.TensorShape.BatchSize())
        throw std::runtime_error("Multiply - batch size mismatch");

    const auto numberSystem = inputA.NumericType;
    const auto deviceType = inputA.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorMul<float>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorMul<float, true>(inputA, inputB, output);
        else
            throw std::runtime_error("Not implemented");
    }
    else
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorMul<int>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorMul<int, true>(inputA, inputB, output);
        else
            throw std::runtime_error("Not implemented");
    }
}

void Add(const Tensor& inputA, const Tensor& inputB,
         Tensor& output)
{
    if (inputA.NumericType != inputB.NumericType ||
        inputA.NumericType != output.NumericType)
        throw std::runtime_error(
            "Add - Number system mismatches between tensors");

    if (inputA.Device != inputB.Device ||
        inputB.Device != output.Device)
        throw std::runtime_error("Add - Device mismatch");

    if (inputA.TensorShape != inputB.TensorShape ||
        inputB.TensorShape != output.TensorShape)
        throw std::runtime_error("Add - Tensor shape mismatch");

    if (inputA.TensorShape.BatchSize() != inputB.TensorShape.BatchSize() ||
        inputB.TensorShape.BatchSize() != output.TensorShape.BatchSize())
        throw std::runtime_error("Add - Batch size mismatch");

    const auto numberSystem = inputA.NumericType;
    const auto deviceType = inputA.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorAdd<float>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorAdd<float, true>(inputA, inputB, output);
        else
            throw std::runtime_error("Not implemented");
    }
    else
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorAdd<int>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorAdd<int, true>(inputA, inputB, output);
        else
            throw std::runtime_error("Not implemented");
    }
}

void Transpose(const Tensor& input, Tensor& output)
{
    if (input.NumericType != output.NumericType)
        throw std::runtime_error("Transpose - Number system mismatch");
    auto transposedInputShape = input.TensorShape;
    transposedInputShape.Transpose();

    if (input.Device != output.Device)
        throw std::runtime_error("Transpose - Device mismatch");

    if (transposedInputShape != output.TensorShape)
        throw std::runtime_error("Transpose - Tensor shape mismatch");

    if (input.TensorShape.BatchSize() != output.TensorShape.BatchSize())
        throw std::runtime_error("Transpose - Batch size mismatch");

    const auto numberSystem = input.NumericType;
    const auto device = input.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (device.Type() == DeviceType::Cpu)
            Native::TensorTranspose<float>(input, output);
        else if (device.Type() == DeviceType::Blaze)
            Blaze::TensorTranspose<float, true>(input, output);
        else
            throw std::runtime_error("Not implemented");
    }
    else
    {
        if (device.Type() == DeviceType::Cpu)
            Native::TensorTranspose<int>(input, output);
        else if (device.Type() == DeviceType::Blaze)
            Blaze::TensorTranspose<int, true>(input, output);
        else
            throw std::runtime_error("Not implemented");
    }
}

void Dot(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    if (inputA.NumericType != inputB.NumericType ||
        inputA.NumericType != output.NumericType)
        throw std::runtime_error(
            "Dot - Number system mismatches between tensors");

    if (inputA.Device != inputB.Device || inputB.Device != output.Device)
        throw std::runtime_error("Dot - Device mismatch");

    if (inputA.TensorShape != inputB.TensorShape ||
        inputB.TensorShape != output.TensorShape)
        throw std::runtime_error("Dot - Tensor shape mismatch");

    if (inputA.TensorShape.BatchSize() != inputB.TensorShape.BatchSize() ||
        inputB.TensorShape.BatchSize() != output.TensorShape.BatchSize())
        throw std::runtime_error("Dot - Batch size mismatch");

    const auto numberSystem = inputA.NumericType;
    const auto device = inputA.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (device.Type() == DeviceType::Cpu || device.Type() ==
            DeviceType::Blaze)
            Native::TensorDot<float>(inputA, inputB, output);
        else
            throw std::runtime_error("Not implemented");
    }
    else
    {
        if (device.Type() == DeviceType::Cpu || device.Type() ==
            DeviceType::Blaze)
            Native::TensorDot<int>(inputA, inputB, output);
        else
            throw std::runtime_error("Not implemented");
    }
}

// void ActivationForward(const Tensor& input, Tensor& output,
//                        std::unique_ptr<ActivationFunc>& activation)
// {
// }
//
// void ActivationBackward(const Tensor& input, Tensor& output,
//                         std::unique_ptr<ActivationFunc>& activation)
// {
// }
//
// void Dot(const Tensor& inputA, const Tensor& inputB, Tensor& output)
// {
// }
} // namespace CubbyDNN
