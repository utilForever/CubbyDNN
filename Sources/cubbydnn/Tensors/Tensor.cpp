/** Copyright (c) 2019 Chris Ohk, Justin Kim
 *
 * We are making my contributions/submissions to this project solely in our
 * personal capacity and are not conveying any rights to any intellectual
 * property of any third parties.
 */

#include <cubbydnn/Tensors/Tensor.hpp>

namespace CubbyDNN
{
Tensor::Tensor(std::unique_ptr<void> Data, TensorInfo info)
    : DataPtr(std::move(Data)), Info(std ::move(info))
{
}

Tensor::Tensor(Tensor&& tensor) noexcept
    : DataPtr(std::move(tensor.DataPtr)), Info(std::move(tensor.Info))
{
}

Tensor& Tensor::operator=(Tensor&& tensor) noexcept
{
    DataPtr = std::move(tensor.DataPtr);
    Info = tensor.Info;
    return *this;
}

Tensor AllocateTensor(const TensorInfo& info)
{
    auto byteSize = info.ByteSize();
    auto dataPtr = std::make_unique<void>(byteSize);
    return Tensor(std::move(dataPtr), info);
}

}  // namespace CubbyDNN