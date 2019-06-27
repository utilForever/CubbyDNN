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

static TensorPtr AllocateTensor(TensorInfo info){
    auto byteSize = info.ByteSize();
    auto dataPtr = std::make_unique<void>(byteSize);
    return std::make_unique<Tensor>(std::move(dataPtr), info);
}

}  // namespace CubbyDNN