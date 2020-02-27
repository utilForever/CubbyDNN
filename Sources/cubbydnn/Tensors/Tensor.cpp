/** Copyright (c) 2019 Chris Ohk, Justin Kim
 *
 * We are making my contributions/submissions to this project solely in our
 * personal capacity and are not conveying any rights to any intellectual
 * property of any third parties.
 */

#include <cubbydnn/Tensors/Tensor.hpp>
#include <iostream>

namespace CubbyDNN
{
Tensor::Tensor(void* Data, TensorInfo info)
    : DataPtr(Data),
      Info(info)
{
    Data = nullptr;
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

size_t Tensor::GetElementOffset(Shape offsetInfo) const
{
    const auto [batchIdx, channelIdx, rowIdx, colIdx] = offsetInfo;
    const auto& shape = Info.GetShape();
    const auto rowSize = shape.Row;
    const auto channelSize = shape.Channel;
    const auto colSize = shape.Col;

    assert(rowIdx < rowSize);
    assert(channelIdx < channelSize);
    assert(colIdx < colSize);

    size_t offset = 0;
    offset += colIdx;
    size_t multiplier = colSize;
    offset += multiplier * rowIdx;
    multiplier *= rowSize;
    offset += multiplier * channelIdx;
    multiplier *= channelSize;
    offset += multiplier * batchIdx;

    return offset;
}

Tensor AllocateTensor(const TensorInfo& info)
{
    const auto byteSize = info.GetByteSize();
    void* dataPtr = static_cast<void*>(malloc(byteSize));
    std::memset(dataPtr, 0, byteSize);
    return Tensor(dataPtr, info);
}

void CopyTensor(Tensor& source, Tensor& destination)
{
    assert(source.Info == destination.Info);
    assert(source.Info.GetByteSize() == destination.Info.GetByteSize());
    const size_t ByteSize = source.Info.GetByteSize();

    std::memcpy(destination.DataPtr, source.DataPtr, ByteSize);
}
} // namespace CubbyDNN
