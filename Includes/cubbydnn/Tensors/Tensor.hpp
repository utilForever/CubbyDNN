// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_DATA_HPP
#define CUBBYDNN_TENSOR_DATA_HPP

#include <cubbydnn/Tensors/TensorInfo.hpp>
#include <cstring>
#include <memory>

namespace CubbyDNN
{
//! TensorData class contains data vector for processing
//! with attributes describing it
//! \tparam T : type of data this tensorData contains
struct Tensor
{
    Tensor() = default;
    Tensor(void* Data, Shape shape, NumberSystem numberSystem);
    ~Tensor();

    Tensor(const Tensor& tensor) = delete;
    Tensor(Tensor&& tensor) noexcept;
    /// move assignment operator
    Tensor& operator=(const Tensor& tensor) = delete;
    Tensor& operator=(Tensor&& tensor) noexcept;

    static void CopyTensor(Tensor& source, Tensor& destination);
    /// Data vector which possesses actual data
    void* DataPtr = nullptr;
    /// Shape of this tensorData
    Shape TensorShape;
    NumberSystem NumericType = NumberSystem::Float;
};


//! Builds empty Tensor so data can be put potentially
//! \param shape : shape of tensor to allocate
//! \param numberSystem : number system of the tensor
//! \return : Tensor that has been allocated
Tensor CreateTensor(const Shape& shape, NumberSystem numberSystem);

template <typename T>
void* AllocateData(const Shape& shape)
{
    const auto byteSize = shape.TotalSize() * sizeof(T);
    void* dataPtr = malloc(byteSize);
    if (dataPtr != nullptr)
        std::memset(dataPtr, 0, byteSize);
    else
        std::runtime_error("Data allocation has failed");
    return dataPtr;
}

//! Used only for testing
template <typename T>
void SetData(std::initializer_list<std::size_t> index, Tensor& tensor, T value)
{
    const auto offset = tensor.TensorShape.Offset(index);
    *(static_cast<T*>(tensor.DataPtr) + offset) = value;
}

//! Used only for testing
template <typename T>
void SetData(std::initializer_list<std::size_t> index, const Shape& shape, void* dataPtr,
             T data)
{
    std::size_t offset = shape.Offset(index);
    *(static_cast<T*>(dataPtr) + offset) = data;
}

//! Used only for testing
template <typename T>
T GetData(std::initializer_list<std::size_t> index, const Tensor& tensor)
{
    const auto offset = tensor.TensorShape.Offset(index);
    return *(static_cast<T*>(tensor.DataPtr) + offset);
}
} // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_DATA_HPP
