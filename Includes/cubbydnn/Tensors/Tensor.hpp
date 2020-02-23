// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_DATA_HPP
#define CUBBYDNN_TENSOR_DATA_HPP

#include <cubbydnn/Tensors/TensorInfo.hpp>

#include <atomic>
#include <bitset>
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>

namespace CubbyDNN
{
//! TensorData class contains data vector for processing
//! with attributes describing it
//! \tparam T : type of data this tensorData contains
struct Tensor
{
    Tensor(void* Data, TensorInfo info);

    ~Tensor()
    {
        free(DataPtr);
    }

    Tensor(Tensor&& tensor) noexcept;
    /// move assignment operator
    Tensor& operator=(Tensor&& tensor) noexcept;

    [[nodiscard]] size_t GetElementOffset(Shape offsetInfo) const;
    /// Data vector which possesses actual data
    void* DataPtr;
    /// Shape of this tensorData
    TensorInfo Info;
};

void CopyTensor(Tensor& source, Tensor& destination);

//! Builds empty Tensor so data can be put potentially
//! \param info : information of tensor to allocate
//! \return : Tensor that has been allocated
Tensor AllocateTensor(const TensorInfo& info);

template <typename T>
void* AllocateData(const Shape& shape)
{
    const auto byteSize =
        shape.Batch * shape.Channel * shape.Row * shape.Col * sizeof(T);
    void* dataPtr = malloc(byteSize);
    std::memset(dataPtr, 0, byteSize);
    return dataPtr;
}

//! Used only for testing
template <typename T>
void SetData(const Shape& index, Tensor& tensor, T value)
{
    const auto offset = tensor.GetElementOffset(index);
    *(static_cast<T*>(tensor.DataPtr) + offset) = value;
}

//! Used only for testing
template <typename T>
void SetData(const Shape& index, const Shape& shape, void* dataPtr, T data)
{
    size_t offset = 0;
    offset += index.Col;
    size_t multiplier = shape.Col;
    offset += multiplier * index.Row;
    multiplier *= shape.Row;
    offset += multiplier * index.Channel;
    multiplier *= shape.Channel;
    offset += multiplier * index.Batch;

    *(static_cast<T*>(dataPtr) + offset) = data;
}

template <typename T>
T GetData(const Shape& index, const Tensor& tensor)
{
    const auto offset = tensor.GetElementOffset(index);
    return *(static_cast<T*>(tensor.DataPtr) + offset);
}
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_DATA_HPP
