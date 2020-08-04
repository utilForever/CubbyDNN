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
Tensor<T>::Tensor(Shape shape, std::size_t batchSize, Compute::Device device)
    : TensorShape(std::move(shape)),
      Device(std::move(device)),
      BatchSize(batchSize)
{
    m_paddedColumnSize = m_getPaddedColumnSize();
    const auto totalSize =
        (TensorShape.Size() / TensorShape.NumCols()) * m_paddedColumnSize;

    T* ptr = new T[totalSize];
    for (std::size_t i = 0; i < totalSize; ++i)
        *(ptr + i) = 0;
    Data = Utils::Span<T>(ptr, totalSize);

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
    const auto totalSize =
        (TensorShape.Size() / TensorShape.NumCols()) * m_paddedColumnSize;
    void* ptr = static_cast<void*>(new float[totalSize]);

    for (std::size_t i = 0; i < TensorShape.Size() / TensorShape.NumCols(); ++i)
        for (std::size_t j = 0; j < m_paddedColumnSize; ++j)
        {
            const auto index = m_paddedColumnSize * i + j;
            *(static_cast<float*>(ptr) + index) = data.at(index);
        }

    Data = Utils::Span<T>(ptr, totalSize);
    m_hasOwnership.exchange(true, std::memory_order_release);
}


// Perform deep copy
template <typename T>
Tensor<T>::Tensor(const Tensor<T>& tensor)
    : TensorShape(tensor.TensorShape),
      Device(tensor.Device),
      BatchSize(tensor.BatchSize)
{
    if (tensor.m_hasOwnership)
    {
        auto dataSize = GetBatchElementSize();

        if (!m_hasOwnership)
        {
            dataSize *= sizeof(T);
            Data = Utils::Span<T>(new float[dataSize], dataSize);
        }
        Utils::Span<T>::DeepCopy(Data, tensor.Data);
        m_hasOwnership.exchange(true, std::memory_order_release);
    }
}

template <typename T>
Tensor<T>::~Tensor()
{
    m_freeData();
}

template <typename T>
Tensor<T>::Tensor(Tensor&& tensor) noexcept
    : Data(tensor.Data),
      TensorShape(std::move(tensor.TensorShape)),
      Device(std::move(tensor.Device)),
      BatchSize(tensor.BatchSize)

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

    if (tensor.m_hasOwnership)
    {
        const auto elementSize = GetBatchElementSize();
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
    T& val = Data.At(offset + GetElementSize() * batchIdx);
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
    const auto numRows = sourceShape.NumRows();
    const auto numCols = sourceShape.NumCols();
    const auto sourceColSize = source.GetColumnElementSize();
    const auto destColSize = destination.GetColumnElementSize();

    const auto elementSize = destination.GetBatchElementSize();
    if (!destination.m_hasOwnership)
    {
        destination.Data = Utils::Span<T>(new T[elementSize], elementSize);
    }

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
} // namespace Takion
#endif
