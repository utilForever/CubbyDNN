// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cassert>
#include <cubbydnn/Tensors/Tensor.hpp>
#include <iostream>

namespace CubbyDNN
{
Tensor::Tensor(Shape shape, Compute::Device device, NumberSystem numberSystem)
    : TensorShape(std::move(shape)),
      NumericType(numberSystem),
      Device(std::move(device))
{
    paddedColumnSize = m_getPaddedColumnSize();
    const auto totalSize = (TensorShape.Size() / TensorShape.NumCols()) *
                           paddedColumnSize;

    if (NumericType == NumberSystem::Float)
    {
        DataPtr = static_cast<void*>(new float[totalSize]);
        for (std::size_t i = 0; i < totalSize; ++i)
            *(static_cast<float*>(DataPtr) + i) = 0;
    }
    else if (NumericType == NumberSystem::Int)
    {
        DataPtr = static_cast<void*>(new int[totalSize]);
        for (std::size_t i = 0; i < totalSize; ++i)
            *(static_cast<float*>(DataPtr) + i) = 0;
    }
}

Tensor::~Tensor()
{
    if (NumericType == NumberSystem::Float)
        delete[] static_cast<float*>(DataPtr);
    else
        delete[] static_cast<int*>(DataPtr);
}

Tensor::Tensor(const Tensor& tensor)
    : DataPtr(tensor.DataPtr),
      TensorShape(tensor.TensorShape),
      NumericType(tensor.NumericType),
      Device(tensor.Device)
{
}

Tensor::Tensor(Tensor&& tensor) noexcept
    : DataPtr(tensor.DataPtr),
      TensorShape(std::move(tensor.TensorShape)),
      NumericType(tensor.NumericType),
      Device(std::move(tensor.Device))
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
        source.Device.PadSize() > 0
            ? source.Device.PadSize()
            : sourceShape.NumCols();
    const auto destColSize =
        destination.Device.PadSize() > 0
            ? destination.Device.PadSize()
            : destShape.NumCols();
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
