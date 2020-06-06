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
// void MultiplyAdd(const Tensor& inputA, const Tensor& inputB,
//                  const Tensor& inputC, Tensor& output, bool transposeA,
//                  bool transposeB)
// {
// }
//
// void BatchMultiply(const Tensor& inputA, const Tensor& batchedInputB,
//                    Tensor& output, bool transposeA,
//                    bool transposeB)
// {
// }

void Multiply(Tensor& inputA, Tensor& inputB,
              Tensor& output, bool transposeA, bool transposeB)
{
    if (inputA.NumericType != inputB.NumericType ||
        inputA.NumericType != output.NumericType)
        throw std::invalid_argument(
            "Multiply - Number system mismatches between tensors");

    if (inputA.Device != inputB.Device ||
        inputB.Device != output.Device)
        throw std::invalid_argument(
            "Multiply - Device Type mismatches between tensors");

    const auto inputShapeA = transposeA
                                 ? inputA.TensorShape.GetTransposedShape()
                                 : inputA.TensorShape;
    const auto inputShapeB = transposeB
                                 ? inputB.TensorShape.GetTransposedShape()
                                 : inputB.TensorShape;

    if (inputShapeA.NumCols() != inputShapeB.NumRows() ||
        inputShapeA.NumRows() != output.TensorShape.NumRows() ||
        inputShapeB.NumCols() != output.TensorShape.NumCols())
        throw std::invalid_argument("Multiply - Tensor shape mismatch");

    if (inputShapeA.NumMatrices() != inputShapeB.NumMatrices() ||
        inputShapeB.NumMatrices() != output.TensorShape.NumMatrices())
        throw std::invalid_argument("Multiply - batch size mismatch");

    const auto numberSystem = inputA.NumericType;
    const auto deviceType = inputA.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorMul<float>(inputA, inputB, output, transposeA,
                                     transposeB);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorMul<float>(inputA, inputB, output, transposeA,
                                    transposeB);
        else
            throw std::invalid_argument("Not implemented");
    }
    else
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorMul<int>(inputA, inputB, output, transposeA,
                                   transposeB);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorMul<int>(inputA, inputB, output, transposeA,
                                  transposeB);
        else
            throw std::invalid_argument("Not implemented");
    }
}

void BatchMean(Tensor& tensor, Tensor& output, std::size_t idx)
{
    if (tensor.NumericType != output.NumericType)
        throw std::invalid_argument(
            "BatchMean - Number system mismatches between tensors");

    if (tensor.Device != output.Device)
        throw std::invalid_argument("BatchMean = Device mismatch");

    std::size_t i = 0;
    while (tensor.TensorShape.Dim() - i != idx)
    {
        const auto tensorIdx = tensor.TensorShape.Dim() - i;
        const auto outputIdx = output.TensorShape.Dim() - i;
        if (tensor.TensorShape.At(tensorIdx) !=
            output.TensorShape.At(outputIdx))
            throw std::invalid_argument("BatchMean - Shape mismatch");
        ++i;
    }

    if (tensor.NumericType == NumberSystem::Float)
    {
        if (tensor.Device.Type() == DeviceType::Blaze)
            Blaze::BatchMean<float>(tensor, idx, output);
        else
            Native::BatchMean<float>(tensor, idx, output);
    }
    else
    {
        if (tensor.Device.Type() == DeviceType::Blaze)
            Blaze::BatchMean<int>(tensor, idx, output);
        else
            Native::BatchMean<int>(tensor, idx, output);
    }
}

void Add(const Tensor& inputA, const Tensor& inputB,
         Tensor& output)
{
    if (inputA.NumericType != inputB.NumericType ||
        inputA.NumericType != output.NumericType)
        throw std::invalid_argument(
            "Add - Number sysconst Tensor&tem mismatches between tensors");

    if (inputA.Device != inputB.Device ||
        inputB.Device != output.Device)
        throw std::invalid_argument("Add - Device mismatch");

    if (inputA.TensorShape != inputB.TensorShape ||
        inputB.TensorShape != output.TensorShape)
        throw std::invalid_argument("Add - Tensor shape mismatch");

    if (inputA.TensorShape.NumMatrices() != inputB.TensorShape.NumMatrices() ||
        inputB.TensorShape.NumMatrices() != output.TensorShape.NumMatrices())
        throw std::invalid_argument("Add - Batch size mismatch");

    const auto numberSystem = inputA.NumericType;
    const auto deviceType = inputA.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorAdd<float>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorAdd<float>(inputA, inputB, output);
        else
            throw std::invalid_argument("Not implemented");
    }
    else
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorAdd<int>(inputA, inputB, output);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorAdd<int>(inputA, inputB, output);
        else
            throw std::invalid_argument("Not implemented");
    }
}

void Add(Tensor& tensor, const Tensor& toAdd)
{
    if (toAdd.NumericType != tensor.NumericType)
        throw std::invalid_argument(
            "Add - Number system mismatches between tensors");

    if (toAdd.Device != tensor.Device)
        throw std::invalid_argument("Add - Device mismatch");

    if (toAdd.TensorShape != tensor.TensorShape)
        throw std::invalid_argument("Add - Tensor shape mismatch");

    if (toAdd.TensorShape.NumMatrices() != tensor.TensorShape.NumMatrices())
        throw std::invalid_argument("Add - Batch size mismatch");

    const auto numberSystem = toAdd.NumericType;
    const auto deviceType = toAdd.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorAdd<float>(tensor, toAdd);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorAdd<float>(tensor, toAdd);
        else
            throw std::invalid_argument("Not implemented");
    }
    else
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorAdd<int>(tensor, toAdd);
        else if (deviceType.Type() == DeviceType::Blaze)
            Blaze::TensorAdd<int>(tensor, toAdd);
        else
            throw std::invalid_argument("Not implemented");
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
        throw std::invalid_argument("Transpose - Number system mismatch");
    auto transposedInputShape = input.TensorShape;
    transposedInputShape.Transpose();

    if (input.Device != output.Device)
        throw std::invalid_argument("Transpose - Device mismatch");

    if (transposedInputShape != output.TensorShape)
        throw std::invalid_argument("Transpose - Tensor shape mismatch");

    if (input.TensorShape.NumMatrices() != output.TensorShape.NumMatrices())
        throw std::invalid_argument("Transpose - Batch size mismatch");

    const auto numberSystem = input.NumericType;
    const auto device = input.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (device.Type() == DeviceType::Cpu)
            Native::TensorTranspose<float>(input, output);
        else if (device.Type() == DeviceType::Blaze)
            Blaze::TensorTranspose<float>(input, output);
        else
            throw std::invalid_argument("Not implemented");
    }
    else
    {
        if (device.Type() == DeviceType::Cpu)
            Native::TensorTranspose<int>(input, output);
        else if (device.Type() == DeviceType::Blaze)
            Blaze::TensorTranspose<int>(input, output);
        else
            throw std::invalid_argument("Not implemented");
    }
}

void Dot(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    if (inputA.NumericType != inputB.NumericType ||
        inputA.NumericType != output.NumericType)
        throw std::invalid_argument(
            "Dot - Number system mismatches between tensors");

    if (inputA.Device != inputB.Device || inputB.Device != output.Device)
        throw std::invalid_argument("Dot - Device mismatch");

    if (inputA.TensorShape != inputB.TensorShape ||
        inputB.TensorShape != output.TensorShape)
        throw std::invalid_argument("Dot - Tensor shape mismatch");

    if (inputA.TensorShape.NumMatrices() != inputB.TensorShape.NumMatrices() ||
        inputB.TensorShape.NumMatrices() != output.TensorShape.NumMatrices())
        throw std::invalid_argument("Dot - Batch size mismatch");

    const auto numberSystem = inputA.NumericType;
    const auto device = inputA.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (device.Type() == DeviceType::Cpu || device.Type() ==
            DeviceType::Blaze)
            Native::TensorDot<float>(inputA, inputB, output);
        else
            throw std::invalid_argument("Not implemented");
    }
    else
    {
        if (device.Type() == DeviceType::Cpu || device.Type() ==
            DeviceType::Blaze)
            Native::TensorDot<int>(inputA, inputB, output);
        else
            throw std::invalid_argument("Not implemented");
    }
}

void ScalarMul(const Tensor& input, Tensor& output, float toMul)
{
    if (input.NumericType != output.NumericType)
        throw std::invalid_argument(
            "ScalarMul - Numeric type mismatch");

    if (input.TensorShape.NumMatrices() != output.TensorShape.NumMatrices())
        throw std::invalid_argument("ScalarMul - Batch size  mismatch");

    Native::ScalarMul<float>(input, output, toMul);
}

void ScalarMul(Tensor& tensor, float toMul)
{
    Native::ScalarMul<float>(tensor, toMul);
}
} // namespace CubbyDNN
