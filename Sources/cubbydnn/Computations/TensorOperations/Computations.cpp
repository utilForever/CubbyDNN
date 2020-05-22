// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Computations/TensorOperations/Computations.hpp>
#include <cubbydnn/Computations/TensorOperations/Native.hpp>
#include <cubbydnn/Computations/TensorOperations/Blaze.hpp>
#include <cubbydnn/Computations/Initializers/InitializerType.hpp>

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

    if (inputA.TensorShape.NumMatrices() != inputB.TensorShape.NumMatrices() ||
        inputB.TensorShape.NumMatrices() != output.TensorShape.NumMatrices())
        throw std::runtime_error("Multiply - batch size mismatch");

    const auto numberSystem = inputA.NumericType;
    const auto deviceType = inputA.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorMul<float>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorMul<float>(inputA, inputB, output);
        else
            throw std::runtime_error("Not implemented");
    }
    else
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorMul<int>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorMul<int>(inputA, inputB, output);
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

    if (inputA.TensorShape.NumMatrices() != inputB.TensorShape.NumMatrices() ||
        inputB.TensorShape.NumMatrices() != output.TensorShape.NumMatrices())
        throw std::runtime_error("Add - Batch size mismatch");

    const auto numberSystem = inputA.NumericType;
    const auto deviceType = inputA.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorAdd<float>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorAdd<float>(inputA, inputB, output);
        else
            throw std::runtime_error("Not implemented");
    }
    else
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorAdd<int>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorAdd<int>(inputA, inputB, output);
        else
            throw std::runtime_error("Not implemented");
    }
}

void Add(Tensor& tensor, const Tensor& toAdd)
{
    if (toAdd.NumericType != tensor.NumericType)
        throw std::runtime_error(
            "Add - Number system mismatches between tensors");

    if (toAdd.Device != tensor.Device)
        throw std::runtime_error("Add - Device mismatch");

    if (toAdd.TensorShape != tensor.TensorShape)
        throw std::runtime_error("Add - Tensor shape mismatch");

    if (toAdd.TensorShape.NumMatrices() != tensor.TensorShape.NumMatrices())
        throw std::runtime_error("Add - Batch size mismatch");

    const auto numberSystem = toAdd.NumericType;
    const auto deviceType = toAdd.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorAdd<float>(tensor, toAdd);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorAdd<float>(tensor, toAdd);
        else
            throw std::runtime_error("Not implemented");
    }
    else
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorAdd<int>(tensor, toAdd);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorAdd<int>(tensor, toAdd);
        else
            throw std::runtime_error("Not implemented");
    }
}

void Add(const std::vector<Tensor>& tensorVector, Tensor& output)
{
    if (tensorVector.empty())
        return;

    const Zeros zeroInitializer;
    zeroInitializer.Initialize(output);

    Tensor::CopyTensor(tensorVector.at(0), output);

    for (const auto& tensor : tensorVector)
        Add(output, tensor);
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

    if (input.TensorShape.NumMatrices() != output.TensorShape.NumMatrices())
        throw std::runtime_error("Transpose - Batch size mismatch");

    const auto numberSystem = input.NumericType;
    const auto device = input.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (device.Type() == DeviceType::Cpu)
            Native::TensorTranspose<float>(input, output);
        else if (device.Type() == DeviceType::Blaze)
            Blaze::TensorTranspose<float>(input, output);
        else
            throw std::runtime_error("Not implemented");
    }
    else
    {
        if (device.Type() == DeviceType::Cpu)
            Native::TensorTranspose<int>(input, output);
        else if (device.Type() == DeviceType::Blaze)
            Blaze::TensorTranspose<int>(input, output);
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

    if (inputA.TensorShape.NumMatrices() != inputB.TensorShape.NumMatrices() ||
        inputB.TensorShape.NumMatrices() != output.TensorShape.NumMatrices())
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
} // namespace CubbyDNN
