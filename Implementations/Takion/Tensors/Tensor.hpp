// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_TENSOR_HPP
#define TAKION_TENSOR_HPP

#include <cstring>
#include <Takion/Tensors/TensorDecl.hpp>
#include <iostream>

namespace Takion
{
template <typename T>
Tensor<T>::Tensor(Shape shape, Compute::Device device)
    : TensorShape(std::move(shape)),
      Device(std::move(device)),
      BatchSize(1)
{
    m_paddedColumnSize = m_getPaddedColumnSize();
    m_elementSize = m_getElementSize();

    T* ptr = new T[m_elementSize];
    for (std::size_t i = 0; i < m_elementSize; ++i)
        *(ptr + i) = 0;
    Data = Utils::Span<T>(ptr, m_elementSize);

    m_hasOwnership.exchange(true, std::memory_order_release);
}


template <typename T>
Tensor<T>::Tensor(Shape shape, std::size_t batchSize, Compute::Device device)
    : TensorShape(std::move(shape)),
      Device(std::move(device)),
      BatchSize(batchSize)
{
    m_paddedColumnSize = m_getPaddedColumnSize();
    m_elementSize = m_getElementSize();

    T* ptr = new T[m_elementSize];
    for (std::size_t i = 0; i < m_elementSize; ++i)
        *(ptr + i) = 0;
    Data = Utils::Span<T>(ptr, m_elementSize);

    m_hasOwnership.exchange(true, std::memory_order_release);
}

template <typename T>
Tensor<T>::Tensor(Shape shape, std::size_t batchSize, Compute::Device device,
                  std::vector<T> data)
    : TensorShape(std::move(shape)),
      Device(std::move(device)),
      BatchSize(batchSize)
{
    m_paddedColumnSize = m_getPaddedColumnSize();
    const auto totalSize = m_getElementSize();
    const auto numRow = TensorShape.NumRow();
    const auto numCol = TensorShape.NumCol();

    T* ptr = new T[m_elementSize];

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t rowIdx = 0; rowIdx < TensorShape.NumRow(); ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < TensorShape.NumCol();
                 ++colIdx)
                this->At(batchIdx, { rowIdx, colIdx }) =
                    data[batchIdx * numRow * numCol + rowIdx * numCol + colIdx];
    }

    Data = Utils::Span<T>(ptr, totalSize);
    m_hasOwnership.exchange(true, std::memory_order_release);
}

template <typename T>
Tensor<T>::~Tensor()
{
    m_freeData();
}

// Performs deep copy
template <typename T>
Tensor<T>::Tensor(const Tensor<T>& tensor)
    : TensorShape(tensor.TensorShape),
      Device(tensor.Device),
      BatchSize(tensor.BatchSize)
{
    if (tensor.m_hasOwnership)
    {
        auto dataSize = BatchElementSize();

        if (!m_hasOwnership)
        {
            dataSize *= sizeof(T);
            Data = Utils::Span<T>(new float[dataSize], dataSize);
        }
        Utils::Span<T>::DeepCopy(Data, tensor.Data);
        m_hasOwnership.exchange(true, std::memory_order_release);
    }
}

//! Performs shallow copy
template <typename T>
Tensor<T>::Tensor(Tensor&& tensor) noexcept
    : Data(tensor.Data),
      TensorShape(std::move(tensor.TensorShape)),
      Device(std::move(tensor.Device)),
      BatchSize(tensor.BatchSize),
      m_elementSize(tensor.m_elementSize),
      m_paddedColumnSize(tensor.m_paddedColumnSize)
{
    if (tensor.m_hasOwnership)
    {
        tensor.m_hasOwnership.exchange(false, std::memory_order_acquire);
        Data = tensor.Data;
        tensor.Data.Clear();
        m_hasOwnership.exchange(true, std::memory_order_release);
    }
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& tensor)
{
    if (this == &tensor)
        return *this;

    TensorShape = tensor.TensorShape;
    Device = tensor.Device;
    BatchSize = tensor.BatchSize;
    m_elementSize = tensor.m_elementSize;
    m_paddedColumnSize = tensor.m_paddedColumnSize;

    if (tensor.m_hasOwnership)
    {
        const auto elementSize = BatchElementSize();
        if (!m_hasOwnership)
        {
            Data = Utils::Span<T>(new T[elementSize], elementSize);
        }

        Utils::Span<T>::DeepCopy(Data, tensor.Data);
        m_hasOwnership.exchange(true, std::memory_order_release);
    }
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& tensor) noexcept
{
    TensorShape = tensor.TensorShape;
    Device = std::move(tensor.Device);
    BatchSize = tensor.BatchSize;
    m_elementSize = tensor.m_elementSize;
    m_paddedColumnSize = tensor.m_paddedColumnSize;

    if (m_hasOwnership)
        m_freeData();

    if (tensor.m_hasOwnership)
    {
        Data = tensor.Data;
        m_hasOwnership.exchange(true, std::memory_order_release);
        tensor.m_hasOwnership.exchange(false, std::memory_order_acquire);
    }
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::SubTensor(std::initializer_list<int> index)
{
    if (index.size() > TensorShape.Dim())
        throw std::invalid_argument(
            "Given index dimension exceeds original dimension");

    auto tensorShapeIndex = index.size() - 1;
    std::vector<std::size_t> subShapeVector(TensorShape.Dim() - index.size());

    std::size_t multiplier = 1;
    for (auto idx = TensorShape.Dim() - 1; idx > tensorShapeIndex; --idx)
    {
        subShapeVector[TensorShape.Dim() - index.size() - 1] = TensorShape[idx];

        if (idx == TensorShape.Dim() - 1)
            multiplier *= ColumnElementSize();
        else
            multiplier *= TensorShape[idx];
    }

    std::size_t offset = 0;
    for (auto itr = index.end(); itr != index.begin();
         --itr, --tensorShapeIndex)
    {
        if (TensorShape[tensorShapeIndex] < *itr)
            throw std::invalid_argument("Given index exceeds original shape");

        offset += multiplier * (*itr);

        multiplier *= TensorShape[tensorShapeIndex];
    }

    auto newTensor = Tensor(subShapeVector, BatchSize, Device);
    const auto newElementSize = newTensor.ElementSize();
    const auto elementSize = ElementSize();

#pragma omp parallel for schedule(static)
    for (std::size_t batchIdx = 0; batchIdx < BatchSize; ++batchIdx)
    {
        for (std::size_t elementIdx = 0; elementIdx < newElementSize;
             ++elementIdx)
        {
            newTensor.Data[batchIdx * newElementSize + elementIdx] =
                Data[batchIdx * elementSize + offset + elementIdx];
        }
    }

    return newTensor;
}

template <typename T>
T& Tensor<T>::At(std::size_t batchIdx, std::vector<std::size_t> index)
{
    if (index.size() != TensorShape.Dim())
        throw std::invalid_argument(
            "Index must have same dimension with tensor shape");
    const auto columnIdx = static_cast<int>(TensorShape.Dim() - 1);
    auto shapeIdx = columnIdx;
    auto idx = columnIdx;
    std::size_t multiplier = 1;
    std::size_t offset = 0;
    for (; shapeIdx >= 0 && idx != static_cast<int>(index.size());
           --shapeIdx, --idx)
    {
        offset += multiplier * index.at(idx);
        if (idx == columnIdx && Device.PadSize() > 0)
            multiplier = m_paddedColumnSize;
        else
            multiplier *= TensorShape.At(idx);
    }
    T& val = Data.At(offset + ElementSize() * batchIdx);
    return val;
}

template <typename T>
void Tensor<T>::ForwardTensorData(Tensor<T>& source, Tensor<T>& destination)
{
    if (source.TensorShape != destination.TensorShape)
        throw std::runtime_error("Information of each tensor should be same");

    if (source.Device == destination.Device)
        MoveTensorData(source, destination);
    else
        CopyTensorData(source, destination);
}

template <typename T>
void Tensor<T>::MoveTensorData(Tensor<T>& source, Tensor<T>& destination)
{
    if (source.Device != destination.Device)
        throw std::invalid_argument(
            "Device type of source and destination tensor must be same when "
            "moving data between tensors");

    if (source.TensorShape != destination.TensorShape)
        throw std::invalid_argument(
            "Shape mismatch between source and destination tensors");

    if (!source.m_hasOwnership)
        throw std::runtime_error(
            "Source tensor does not have ownership of the data");

    //! Deallocate data in destination if it already has allocated data to
    //! prevent memory leaks
    if (destination.m_hasOwnership)
        destination.m_freeData();

    destination.Data = source.Data;
    destination.m_hasOwnership.exchange(true, std::memory_order_release);
    source.m_hasOwnership.exchange(false, std::memory_order_acquire);
}

template <typename T>
void Tensor<T>::CopyTensorData(const Tensor<T>& source, Tensor<T>& destination)
{
    if (source.TensorShape != destination.TensorShape)
        throw std::invalid_argument(
            "Shape mismatch between source and destination tensors");

    if (!source.m_hasOwnership)
        throw std::runtime_error(
            "Source tensor does not have ownership of the data");

    const auto sourceShape = source.TensorShape;
    const auto destShape = destination.TensorShape;
    const auto batchSize = sourceShape.NumMatrices();
    const auto numRows = sourceShape.NumRow();
    const auto numCols = sourceShape.NumCol();
    const auto sourceColSize = source.ColumnElementSize();
    const auto destColSize = destination.ColumnElementSize();

    const auto batchElementSize = destination.BatchElementSize();
    if (!destination.m_hasOwnership)
    {
        destination.Data = Utils::Span<T>(new T[batchElementSize],
                                          batchElementSize);
    }

#pragma omp parallel for schedule(static)
    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t rowIdx = 0; rowIdx < numRows; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < numCols; ++colIdx)
            {
                destination
                    .Data[batchIdx * (destColSize * numRows) +
                          destColSize * rowIdx + colIdx] =
                    source
                    .Data[batchIdx * (sourceColSize * numRows) +
                          sourceColSize * rowIdx + colIdx];
            }
    }
}


template <typename T>
std::size_t Tensor<T>::m_getElementSize() const
{
    std::size_t size = 1;
    for (std::size_t i = 0; i < TensorShape.Dim() - 2; ++i)
    {
        size *= TensorShape.At(i);
    }
    size *= m_paddedColumnSize;
    return size;
}

template <typename T>
std::size_t Tensor<T>::m_getPaddedColumnSize() const
{
    if (Device.PadSize() == 0)
        return TensorShape.NumCol();

    const std::size_t padUnitSize = Device.PadSize() / sizeof(T);

    std::size_t i = 0;
    while (padUnitSize * i < TensorShape.NumCol())
        ++i;

    return padUnitSize * i;
}

template <typename T>
void Tensor<T>::m_freeData()
{
    if (m_hasOwnership)
    {
        m_hasOwnership.exchange(false, std::memory_order_acquire);
        Data.Clear();
    }
}
} // namespace Takion
#endif
