// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cassert>
#include <cubbydnn/Tensors/Tensor.hpp>
#include <iostream>

namespace CubbyDNN
{
Tensor::Tensor(void* Data, Shape shape, NumberSystem numberSystem,
               Compute::Device device)
    : DataPtr(Data),
      TensorShape(std::move(shape)),
      NumericType(numberSystem),
      Device(device)
{
    Data = nullptr;
}

Tensor::~Tensor()
{
    if (NumericType == NumberSystem::Float)
        delete[] static_cast<float*>(DataPtr);
    else
        delete[] static_cast<int*>(DataPtr);
}

Tensor::Tensor(Tensor&& tensor) noexcept
    : DataPtr(tensor.DataPtr),
      TensorShape(std::move(tensor.TensorShape)),
      NumericType(tensor.NumericType),
      Device(tensor.Device)
{
    tensor.DataPtr = nullptr;
}

Tensor& Tensor::operator=(Tensor&& tensor) noexcept
{
    DataPtr = tensor.DataPtr;
    tensor.DataPtr = nullptr;
    TensorShape = tensor.TensorShape;
    NumericType = tensor.NumericType;
    return *this;
}

Tensor Tensor::CreateTensor(const Shape& shape, NumberSystem numberSystem,
                            Compute::Device deviceType, std::size_t padSize)
{
    void* dataPtr = nullptr;
    const auto totalSize =
        padSize > 0
            ? shape.BatchSize() * shape.NumRows() * padSize
            : shape.Size();
    if (numberSystem == NumberSystem::Float)
    {
        dataPtr = static_cast<void*>(new float[shape.Size()]);
        for (std::size_t i = 0; i < totalSize; ++i)
            static_cast<float*>(dataPtr)[i] = 0.0f;
    }
    else if (numberSystem == NumberSystem::Int)
    {
        dataPtr = static_cast<void*>(new int[shape.Size()]);
        for (std::size_t i = 0; i < totalSize; ++i)
            static_cast<int*>(dataPtr)[i] = 0;
    }

    return Tensor(dataPtr, shape, numberSystem, deviceType);
}

void Tensor::CopyTensor(const Tensor& source, Tensor& destination)
{
    if (source.TensorShape != destination.TensorShape)
        throw std::runtime_error("Information of each tensor should be same");
    if (source.NumericType != destination.NumericType)
        throw std::runtime_error("NumberSystem of two tensors does not match");

    const auto sourceShape = source.TensorShape;
    const auto destShape = destination.TensorShape;
    const auto batchSize = sourceShape.BatchSize();
    const auto numRows = sourceShape.NumRows();
    const auto numCols = sourceShape.NumCols();
    const auto sourceColSize =
        source.PadSize > 0 ? source.PadSize : sourceShape.NumCols();
    const auto destColSize =
        destination.PadSize > 0 ? destination.PadSize : destShape.NumCols();
    const NumberSystem numericType = source.NumericType;

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t rowIdx = 0; rowIdx < numRows; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < numCols; ++colIdx)
            {
                if (numericType == NumberSystem::Float)
                    static_cast<float*>(
                            destination.DataPtr)[
                            batchIdx * (destColSize * numRows) +
                            destColSize * rowIdx + colIdx] =
                        static_cast<float*>(
                            source.DataPtr)[
                            batchIdx * (sourceColSize * numRows) +
                            sourceColSize * rowIdx + colIdx];
                else
                    static_cast<int*>(
                            destination
                            .DataPtr)[batchIdx * (destColSize * numRows) +
                                      destColSize * rowIdx + colIdx] =
                        static_cast<int*>(
                            source
                            .DataPtr)[batchIdx * (sourceColSize * numRows) +
                                      sourceColSize * rowIdx + colIdx];
            }
    }
}
} // namespace CubbyDNN
