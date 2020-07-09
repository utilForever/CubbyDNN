// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cassert>
#include <cstring>
#include <cubbydnn/Tensors/Tensor.hpp>
#include <iostream>

namespace CubbyDNN
{
Tensor::Tensor(Shape shape, Compute::Device device, NumberSystem numberSystem)
    : TensorShape(std::move(shape)),
      NumericType(numberSystem),
      Device(std::move(device))
{
    if (numberSystem == NumberSystem::Float)
        numPaddedColumn = m_getPaddedColumnSize();
    else
        numPaddedColumn = m_getPaddedColumnSize();
    const auto totalSize = (TensorShape.Size() / TensorShape.NumCols()) *
                           numPaddedColumn;

    if (NumericType == NumberSystem::Float)
    {
        void* ptr = static_cast<void*>(new float[totalSize]);
        for (std::size_t i = 0; i < totalSize; ++i)
            *(static_cast<float*>(ptr) + i) = 0;
        DataPtr = ptr;
    }
    else if (NumericType == NumberSystem::Int)
    {
        void* ptr = static_cast<void*>(new int[totalSize]);
        for (std::size_t i = 0; i < totalSize; ++i)
            *(static_cast<int*>(ptr) + i) = 0;
        DataPtr = ptr;
    }
}

// Perform deep copy
Tensor::Tensor(const Tensor& tensor)
    : TensorShape(tensor.TensorShape),
      NumericType(tensor.NumericType),
      Device(tensor.Device)
{
    auto dataSize = getDataSize();
    if (NumericType == NumberSystem::Float)
    {
        dataSize *= sizeof(float);
        DataPtr = new float[dataSize];
    }
    else
    {
        dataSize *= sizeof(int);
        DataPtr = new float[dataSize];
    }
    for (std::size_t i = 0; i < dataSize; ++i)
    {
        std::memcpy(DataPtr, tensor.DataPtr, dataSize);
    }
}

Tensor::~Tensor()
{
    if (NumericType == NumberSystem::Float)
        delete[] static_cast<float*>(DataPtr);
    else
        delete[] static_cast<int*>(DataPtr);
}

Tensor::Tensor(Tensor&& tensor) noexcept
    : DataPtr(std::move(tensor.DataPtr)),
      TensorShape(std::move(tensor.TensorShape)),
      NumericType(tensor.NumericType),
      Device(std::move(tensor.Device))
{
}

Tensor& Tensor::operator=(const Tensor& tensor)
{
    auto dataSize = getDataSize();
    if (NumericType == NumberSystem::Float)
    {
        dataSize *= sizeof(float);
        DataPtr = new float[dataSize];
    }
    else
    {
        dataSize *= sizeof(int);
        DataPtr = new float[dataSize];
    }
    for (std::size_t i = 0; i < dataSize; ++i)
    {
        std::memcpy(DataPtr, tensor.DataPtr, dataSize);
    }

    TensorShape = tensor.TensorShape;
    NumericType = tensor.NumericType;
    return *this;
}


Tensor& Tensor::operator=(Tensor&& tensor) noexcept
{
    DataPtr = tensor.DataPtr;
    TensorShape = tensor.TensorShape;
    NumericType = tensor.NumericType;
    return *this;
}


void Tensor::ForwardTensor(const Tensor& source, Tensor& destination)
{
    if (source.TensorShape != destination.TensorShape)
        throw std::runtime_error("Information of each tensor should be same");
    if (source.NumericType != destination.NumericType)
        throw std::runtime_error("NumberSystem of two tensors does not match");

    if (source.Device == destination.Device)
        MoveTensor(source, destination);
    else
        CopyTensor(source, destination);
}

void Tensor::MoveTensor(const Tensor& source, Tensor& destination)
{
    if (source.Device != destination.Device)
        throw std::invalid_argument(
            "Device type of source and destination tensor must be same when "
            "moving between tensors");
    destination.DataPtr = source.DataPtr;
}

void Tensor::CopyTensor(const Tensor& source, Tensor& destination)
{
    const auto numericType = source.NumericType;
    const auto sourceShape = source.TensorShape;
    const auto destShape = destination.TensorShape;
    const auto batchSize = sourceShape.NumMatrices();
    const auto numRows = sourceShape.NumRows();
    const auto numCols = sourceShape.NumCols();
    const auto sourceColSize = source.GetColumnElementSize();
    const auto destColSize = destination.GetColumnElementSize();

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
                            destination.DataPtr)[
                            batchIdx * (destColSize * numRows) +
                            destColSize * rowIdx + colIdx] =
                        static_cast<int*>(
                            source.DataPtr)[
                            batchIdx * (sourceColSize * numRows) +
                            sourceColSize * rowIdx + colIdx];
            }
    }
}
} // namespace CubbyDNN
