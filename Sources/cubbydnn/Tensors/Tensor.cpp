// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#include <cubbydnn/Tensors/Tensor.hpp>
#include <iostream>
#include <cassert>

namespace CubbyDNN
{
Tensor::Tensor(void* Data, Shape shape, NumberSystem numberSystem)
    : DataPtr(Data),
      TensorShape(std::move(shape)),
      NumericType(numberSystem)
{
    Data = nullptr;
}

Tensor::~Tensor()
{
    free(DataPtr);
}

Tensor::Tensor(Tensor&& tensor) noexcept
    : DataPtr(tensor.DataPtr),
      TensorShape(std::move(tensor.TensorShape)),
      NumericType(tensor.NumericType)
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

Tensor CreateTensor(const Shape& shape, NumberSystem numberSystem)
{
    std::size_t byteSize;
    if (numberSystem == NumberSystem::Float)
        byteSize = shape.TotalSize() * sizeof(float);
    else
        byteSize = shape.TotalSize() * sizeof(int);
    void* dataPtr = static_cast<void*>(malloc(byteSize));
    std::memset(dataPtr, 0, byteSize);
    return Tensor(dataPtr, shape, numberSystem);
}

void Tensor::CopyTensor(const Tensor& source, Tensor& destination)
{
    if (source.TensorShape != destination.TensorShape)
        throw std::runtime_error("Information of each tensor should be same");
    if (source.NumericType != destination.NumericType)
        throw std::runtime_error("NumberSystem of two tensors does not match");

    std::size_t byteSize;
    const auto shape = source.TensorShape;
    if (source.NumericType == NumberSystem::Float)
        byteSize = shape.TotalSize() * sizeof(float);
    else
        byteSize = shape.TotalSize() * sizeof(int);

    std::memcpy(destination.DataPtr, source.DataPtr, byteSize);
}
} // namespace CubbyDNN
