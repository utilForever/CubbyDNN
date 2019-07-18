/** Copyright (c) 2019 Chris Ohk, Justin Kim
 *
 * We are making my contributions/submissions to this project solely in our
 * personal capacity and are not conveying any rights to any intellectual
 * property of any third parties.
 */

#include <cubbydnn/Tensors/Tensor.hpp>

namespace CubbyDNN
{
Tensor::Tensor(void* Data, TensorInfo info)
    : DataPtr(Data), Info(std ::move(info))
{
    Data = nullptr;
}

Tensor::Tensor(Tensor&& tensor) noexcept
    : DataPtr(tensor.DataPtr), Info(std::move(tensor.Info))
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
    auto byteSize = info.ByteSize();
    void* dataPtr = (void*)malloc(byteSize);
    return Tensor(dataPtr, info);
}

void CopyTensor(Tensor& source, Tensor& destination)
{
    assert(source.Info == destination.Info);
    assert(source.Info.ByteSize() == destination.Info.ByteSize());
    size_t ByteSize = source.Info.ByteSize();

    std::memcpy(destination.DataPtr, source.DataPtr, ByteSize);
}

}  // namespace CubbyDNN