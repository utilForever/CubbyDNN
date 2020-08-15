// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_TENSOR_HPP
#define TAKION_TENSOR_HPP

#include <cstring>
#include <iostream>
#include <Takion/Tensors/TensorDecl.hpp>

namespace Takion
{
template <typename T>
Tensor<T>::Tensor(Shape shape, Compute::Device device)
    : TensorShape(std::move(shape)),
      Device(std::move(device)),
      BatchSize(1)
{
    m_columnElementSize = m_getPaddedColumnSize();
    m_elementSize = m_getElementSize();
    const auto totalSize = m_elementSize * BatchSize;
    const auto size = TensorShape.Size();

    Data = Utils::Span<T>(new T[totalSize], totalSize);

    for (std::size_t idx = 0; idx < size * BatchSize; ++idx)
    {
        At(idx) = 0;
    }

    m_hasOwnership.exchange(true, std::memory_order_release);
}


template <typename T>
Tensor<T>::Tensor(Shape shape, std::size_t batchSize, Compute::Device device)
    : TensorShape(std::move(shape)),
      Device(std::move(device)),
      BatchSize(batchSize)
{
    if (batchSize == 0)
        throw std::invalid_argument("Batch size must be larger than 0");

    m_columnElementSize = m_getPaddedColumnSize();
    m_elementSize = m_getElementSize();
    const auto totalSize = m_elementSize * BatchSize;
    const auto size = TensorShape.Size();

    Data = Utils::Span<T>(new T[totalSize], totalSize);

    for (std::size_t idx = 0; idx < size * BatchSize; ++idx)
    {
        At(idx) = 0;
    }

    m_hasOwnership.exchange(true, std::memory_order_release);
}

template <typename T>
Tensor<T>::Tensor(Shape shape, std::size_t batchSize, Compute::Device device,
                  std::vector<T> data)
    : TensorShape(std::move(shape)),
      Device(std::move(device)),
      BatchSize(batchSize)
{
    if (batchSize == 0)
        throw std::invalid_argument("Batch size must be larger than 0");

    m_columnElementSize = m_getPaddedColumnSize();
    m_elementSize = m_getElementSize();
    const auto totalSize = m_elementSize * BatchSize;
    const auto size = TensorShape.Size();

    Data = Utils::Span<T>(new T[totalSize], totalSize);


    for (std::size_t idx = 0; idx < size * BatchSize; ++idx)
    {
        At(idx) = data[idx];
    }
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
        auto dataSize = TotalElementSize();

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
      m_columnElementSize(tensor.m_columnElementSize)
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
    m_columnElementSize = tensor.m_columnElementSize;

    Tensor<T>::CopyTensorData(tensor, *this);

    return *this;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& tensor) noexcept
{
    TensorShape = tensor.TensorShape;
    Device = tensor.Device;
    BatchSize = tensor.BatchSize;
    m_elementSize = tensor.m_elementSize;
    m_columnElementSize = tensor.m_columnElementSize;

    Tensor<T>::MoveTensorData(tensor, *this);

    return *this;
}

template <typename T>
void Tensor<T>::SetData(const std::vector<T>& data)
{
    const auto totalSize = m_elementSize * BatchSize;
    const auto size = TensorShape.Size();

    if (m_hasOwnership == false)
    {
        T* ptr = new T[totalSize];
        Data = Utils::Span<T>(ptr, totalSize);
    }

    for (std::size_t idx = 0; idx < size * BatchSize; ++idx)
    {
        At(idx) = data[idx];
    }

    m_hasOwnership.exchange(true, std::memory_order_release);
}

template <typename T>
std::size_t Tensor<T>::NumMatrix() const
{
    return TotalElementSize() / (m_columnElementSize * TensorShape.NumRow());
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
    auto idx = columnIdx;
    std::size_t multiplier = 1;
    std::size_t offset = 0;
    for (; idx >= 0 && idx != static_cast<int>(index.size()); --idx)
    {
        offset += multiplier * index.at(idx);
        if (idx == columnIdx)
            multiplier = m_columnElementSize;
        else
            multiplier *= TensorShape.At(idx);
    }
    T& val = Data.At(offset + m_elementSize * batchIdx);
    return val;
}

template <typename T>
const T& Tensor<T>::At(std::size_t batchIdx,
                       std::vector<std::size_t> index) const
{
    if (index.size() != TensorShape.Dim())
        throw std::invalid_argument(
            "Index must have same dimension with tensor shape");
    const auto columnIdx = static_cast<int>(TensorShape.Dim() - 1);
    auto idx = columnIdx;
    std::size_t multiplier = 1;
    std::size_t offset = 0;
    for (; idx >= 0 && idx != static_cast<int>(index.size()); --idx)
    {
        offset += multiplier * index.at(idx);
        if (idx == columnIdx)
            multiplier = m_columnElementSize;
        else
            multiplier *= TensorShape.At(idx);
    }
    const T& val = Data.At(offset + m_elementSize * batchIdx);
    return val;
}

template <typename T>
T& Tensor<T>::At(std::size_t idx)
{
    if (TensorShape.NumCol() == 0)
        throw std::invalid_argument("Accessing data of empty tensor");

    const auto colIdx = idx / TensorShape.NumCol();
    const auto dataIdx =
        m_columnElementSize * colIdx + idx % TensorShape.NumCol();

    const auto limit = m_elementSize * BatchSize;
    if (dataIdx >= limit)
        throw std::invalid_argument("Idx exceeds allocated size");

    return Data[dataIdx];
}

template <typename T>
const T& Tensor<T>::At(std::size_t idx) const
{
    if (TensorShape.NumCol() == 0)
        throw std::invalid_argument("Accessing data of empty tensor");

    const auto colIdx = idx / TensorShape.NumCol();
    const auto dataIdx =
        m_columnElementSize * colIdx + idx % TensorShape.NumCol();

    const auto limit = m_elementSize * BatchSize;
    if (dataIdx >= limit)
        throw std::invalid_argument("Idx exceeds allocated size");

    return Data[dataIdx];
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

    source.m_hasOwnership.exchange(false, std::memory_order_acquire);
    destination.Data = source.Data;
    destination.m_hasOwnership.exchange(true, std::memory_order_release);
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
    const auto batchElementSize = destination.TotalElementSize();
    const auto unitSize = destination.TensorShape.Size();

    if (!destination.m_hasOwnership)
    {
        destination.Data = Utils::Span<T>(new T[batchElementSize],
                                          batchElementSize);
    }

    const long blockSize = 100;
    const auto loopSize = unitSize * destination.BatchSize;

    for (long blockIdx = 0; static_cast<std::size_t>(blockIdx) < loopSize;
         blockIdx += blockSize)
    {
        const auto boundary = blockIdx + blockSize;
        const auto limit =
            static_cast<long>(loopSize) < boundary
                ? loopSize
                : static_cast<std::size_t>(boundary);
        for (std::size_t idx = blockIdx; idx < limit; ++idx)
            destination.At(idx) = source.At(idx);
    }
}


template <typename T>
std::size_t Tensor<T>::m_getElementSize() const
{
    std::size_t size = 1;
    for (std::size_t i = 0; i < TensorShape.Dim() - 1; ++i)
    {
        size *= TensorShape.At(i);
    }
    size *= m_columnElementSize;
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
