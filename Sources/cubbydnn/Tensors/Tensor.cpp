// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#include <cubbydnn/Tensors/Tensor.hpp>
#include <iostream>
#include <cassert>

namespace CubbyDNN
{
Tensor::Tensor(void* Data, TensorInfo info)
    : DataPtr(Data),
      Info(info)
{
    Data = nullptr;
}

Tensor::~Tensor()
{
    free(DataPtr);
}

Tensor::Tensor(Tensor&& tensor) noexcept
    : DataPtr(tensor.DataPtr),
      Info(tensor.Info)
{
    tensor.DataPtr = nullptr;
}

Tensor& Tensor::operator=(Tensor&& tensor) noexcept
{
    DataPtr = tensor.DataPtr;
    tensor.DataPtr = nullptr;
    Info = tensor.Info;
    return *this;
}

Tensor AllocateTensor(const TensorInfo& info)
{
    const auto byteSize = info.GetByteSize();
    void* dataPtr = static_cast<void*>(malloc(byteSize));
    std::memset(dataPtr, 0, byteSize);
    return Tensor(dataPtr, info);
}

void Tensor::CopyTensor(Tensor& source, Tensor& destination)
{
    if (source.Info != destination.Info)
        throw std::runtime_error("Information of each tensor should be same");
    assert(source.Info.GetByteSize() == destination.Info.GetByteSize());
    const std::size_t ByteSize = source.Info.GetByteSize();

    std::memcpy(destination.DataPtr, source.DataPtr, ByteSize);
}
} // namespace CubbyDNN
