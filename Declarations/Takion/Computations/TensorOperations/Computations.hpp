// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_OPERATIONS_HPP
#define CUBBYDNN_OPERATIONS_HPP

#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Computations/TensorOperations/Native.hpp>
#include <cubbydnn/Computations/Initializers/InitializerType.hpp>


namespace Takion::Compute
{
template <typename T>
void MultiplyAdd(const Tensor<T>& inputA, const Tensor<T>& inputB,
                 const Tensor<T>& inputC, Tensor<T>& output, bool transposeA,
                 bool transposeB);

template <typename T>
void Multiply(const Tensor<T>& inputA, const Tensor<T>& inputB,
              Tensor<T>& output, bool transposeA = false,
              bool transposeB = false, bool broadCast = false)
{
    if (inputA.Device != inputB.Device || inputB.Device != output.Device)
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

    if (broadCast)
    {
        const auto batchSizeA = inputShapeA.NumMatrices();
        const auto batchSizeB = inputShapeB.NumMatrices();
        const auto batchSize =
            batchSizeA > batchSizeB ? batchSizeA : batchSizeB;

        if (output.TensorShape.NumMatrices() != batchSize)
            throw std::invalid_argument(
                "Multiply - Mismatch between maximum input batch size and "
                "output batch size");

        if (batchSize % batchSizeA != 0 || batchSize % batchSizeB != 0)
            throw std::invalid_argument(
                "Multiply - input with larger batch size is not multiple of "
                "input "
                "with smaller batch size - failed to broadcast");
    }
    else
    {
        if (inputShapeA.NumMatrices() != inputShapeB.NumMatrices() ||
            inputShapeA.NumMatrices() != output.TensorShape.NumMatrices())
            throw std::invalid_argument(
                "Multiply - Mismatch between batch size");
    }

    const Zeros zeroInitializer;
    zeroInitializer.Initialize(output);

    const auto deviceType = inputA.Device;

    if (deviceType.Type() == DeviceType::Cpu)
        Native::TensorMul<T>(inputA, inputB, output, transposeA,
                             transposeB);
    else
        throw std::invalid_argument("Not implemented");
}

template <typename T>
void Add(const Tensor<T>& inputA, const Tensor<T>& inputB, Tensor<T>& output,
         bool broadCast = false)
{
    if (inputA.Device != inputB.Device || inputB.Device != output.Device)
        throw std::invalid_argument("Add - Device mismatch");

    if (inputA.TensorShape.NumRows() != inputB.TensorShape.NumRows() ||
        inputA.TensorShape.NumCols() != inputB.TensorShape.NumCols() ||
        inputA.TensorShape.NumRows() != output.TensorShape.NumRows() ||
        inputA.TensorShape.NumCols() != output.TensorShape.NumCols())
        throw std::invalid_argument("Add - Tensor shape mismatch");

    if (broadCast)
    {
        const auto batchSizeA = inputA.TensorShape.NumMatrices();
        const auto batchSizeB = inputB.TensorShape.NumMatrices();
        const auto batchSize =
            batchSizeA > batchSizeB ? batchSizeA : batchSizeB;

        if (output.TensorShape.NumMatrices() != batchSize)
            throw std::invalid_argument(
                "Add - Mismatch between maximum input batch size and output "
                "batch size");

        if (batchSize % batchSizeA != 0 || batchSize % batchSizeB != 0)
            throw std::invalid_argument(
                "Add - input with larger batch size is not multiple of input "
                "with smaller batch size - failed to broadcast");
    }
    else
    {
        if (inputA.TensorShape.NumMatrices() !=
            inputB.TensorShape.NumMatrices() ||
            inputB.TensorShape.NumMatrices() !=
            output.TensorShape.NumMatrices())
            throw std::invalid_argument("Add - Batch size mismatch");
    }

    const auto numberSystem = inputA.NumericType;
    const auto deviceType = inputA.Device;

    if (numberSystem == NumberSystem::Float)
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorAdd<float>(inputA, inputB, output);
        else
            throw std::invalid_argument("Not implemented");
    }
    else
    {
        if (deviceType.Type() == DeviceType::Cpu)
            Native::TensorAdd<int>(inputA, inputB, output);
        else
            throw std::invalid_argument("Not implemented");
    }
}

template <typename T>
void Add(Tensor<T>& tensor, const Tensor<T>& toAdd, bool broadCast = false)
{
    if (toAdd.NumericType != tensor.NumericType)
        throw std::invalid_argument(
            "Add - Number system mismatches between tensors");

    if (toAdd.Device != tensor.Device)
        throw std::invalid_argument("Add - Device mismatch");

    if (toAdd.TensorShape.NumRows() != tensor.TensorShape.NumRows() ||
        toAdd.TensorShape.NumCols() != tensor.TensorShape.NumCols())
        throw std::invalid_argument("Add - Tensor shape mismatch");

    if (broadCast)
    {
        if (tensor.TensorShape.NumMatrices() < toAdd.TensorShape.NumMatrices())
            throw std::invalid_argument(
                "Add - tensor must have equal or larger batch size size than "
                "toAdd");
        if (tensor.TensorShape.NumMatrices() %
            toAdd.TensorShape.NumMatrices() !=
            0)
            throw std::invalid_argument(
                "Add - batch size of given tensor is not multiple of batch "
                "size of tensor to Add - failed to broadcast");
    }
    else
    {
        if (toAdd.TensorShape.NumMatrices() != tensor.TensorShape.NumMatrices())
            throw std::invalid_argument("Add - Batch size mismatch");
    }

    const auto numberSystem = toAdd.NumericType;
    const auto deviceType = toAdd.Device;

    if (deviceType.Type() == DeviceType::Cpu)
        Native::TensorAdd<T>(tensor, toAdd);
    else
        throw std::invalid_argument("Not implemented");
}

//! Adds up all tensors in tensorVector and outputs result to output
// void Add(const std::vector<Tensor>& tensorVector, Tensor& output);

//! Returns mean of input tensor from axis
//! This function assumes the highest dimension is for batch
//! \param idx : index that indicates end of batch
template <typename T>
void Shrink(Tensor<T>& tensor, Tensor<T>& output, int index = -1)
{
    if (tensor.NumericType != output.NumericType)
        throw std::invalid_argument(
            "Shrink - Number system mismatches between tensors");

    if (tensor.Device != output.Device)
        throw std::invalid_argument("Shrink = Device mismatch");

    int idx;
    if (index < 0)
        idx = static_cast<int>(tensor.TensorShape.Dim()) -
              static_cast<int>(output.TensorShape.Dim());
    else
        idx = static_cast<int>(tensor.TensorShape.Dim()) - (index + 1);

    if (idx < 0)
        throw std::invalid_argument("invalid given index to shrink");

    std::size_t i = 1;
    while (tensor.TensorShape.Dim() - i >= static_cast<std::size_t>(idx))
    {
        const auto tensorIdx = tensor.TensorShape.Dim() - i;
        const auto outputIdx = output.TensorShape.Dim() - i;
        if (tensor.TensorShape.At(tensorIdx) !=
            output.TensorShape.At(outputIdx))
            throw std::invalid_argument("Shrink - Shape mismatch");
        ++i;
    }

    const Zeros zeroInitializer;
    zeroInitializer.Initialize(output);
    Native::BatchMean<T>(tensor, idx, output);
}

template <typename T>
void Transpose(const Tensor<T>& input, Tensor<T>& output)
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

    if (device.Type() == DeviceType::Cpu)
        Native::TensorTranspose<T>(input, output);
}

template <typename T>
void Dot(const Tensor<T>& inputA, const Tensor<T>& inputB,
         Tensor<T>& output,
         bool broadCast = false)
{
    if (inputA.NumericType != inputB.NumericType ||
        inputA.NumericType != output.NumericType)
        throw std::invalid_argument(
            "Dot - Number system mismatches between tensors");

    if (inputA.Device != inputB.Device || inputB.Device != output.Device)
        throw std::invalid_argument("Dot - Device mismatch");

    if (inputA.TensorShape.NumRows() != inputB.TensorShape.NumRows() ||
        inputA.TensorShape.NumCols() != inputB.TensorShape.NumCols() ||
        inputA.TensorShape.NumRows() != output.TensorShape.NumRows() ||
        inputA.TensorShape.NumCols() != output.TensorShape.NumCols())
        throw std::invalid_argument("Add - Tensor shape mismatch");

    if constexpr (broadCast)
    {
        const auto batchSizeA = inputA.TensorShape.NumMatrices();
        const auto batchSizeB = inputB.TensorShape.NumMatrices();
        const auto batchSize =
            batchSizeA > batchSizeB ? batchSizeA : batchSizeB;

        if (output.TensorShape.NumMatrices() != batchSize)
            throw std::invalid_argument(
                "Dot - Mismatch between maximum input batch size and output "
                "batch size");

        if (batchSize % batchSizeA != 0 || batchSize % batchSizeB != 0)
            throw std::invalid_argument(
                "Dot - input with larger batch size is not multiple of input "
                "with smaller batch size - failed to broadCast");
    }
    else
    {
        if (inputA.TensorShape.NumMatrices() !=
            inputB.TensorShape.NumMatrices() ||
            inputB.TensorShape.NumMatrices() !=
            output.TensorShape.NumMatrices())
            throw std::invalid_argument("Dot - Batch size mismatch");
    }

    const auto numberSystem = inputA.NumericType;
    const auto device = inputA.Device;

    if (device.Type() == DeviceType::Cpu ||
        device.Type() == DeviceType::Blaze)
        Native::TensorDot<T>(inputA, inputB, output);
    else
        throw std::invalid_argument("Not implemented");
}

template <typename T>
void ScalarMul(const Tensor<T>& input, Tensor<T>& output, float toMul)
{
    if (input.TensorShape.NumMatrices() != output.TensorShape.NumMatrices())
        throw std::invalid_argument("ScalarMul - Batch size  mismatch");

    Native::ScalarMul<T>(input, output, toMul);
}

template <typename T>
void ScalarMul(Tensor<T>& tensor, T toMul)
{
    Native::ScalarMul<T>(tensor, toMul);
}
} // namespace Takion
#endif
