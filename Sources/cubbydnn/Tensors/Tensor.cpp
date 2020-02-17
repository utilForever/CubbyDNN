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
    : DataPtr(Data),
      Info(std::move(info))
{
    Data = nullptr;
}

Tensor::Tensor(Tensor&& tensor) noexcept
    : DataPtr(tensor.DataPtr),
      Info(std::move(tensor.Info))
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

size_t Tensor::GetElementOffset(ShapeOffsetInfo offsetInfo) const
{
    const auto [batchIdx, rowIdx, channelIdx, colIdx] = offsetInfo;
    const auto shapeIndex = Info.GetShapeIndex();
    const auto& shape = Info.GetShape();
    const auto batchSize = shape.at(shapeIndex.BatchSizeIdx);
    const auto rowSize = shape.at(shapeIndex.RowSizeIdx);
    const auto channelSize = shape.at(shapeIndex.ChannelSizeIdx);
    const auto colSize = shape.at(shapeIndex.ColSizeIdx);

    size_t offset = 0;
    size_t multiplier = 1;

    offset += colSize * colIdx;
    multiplier = colSize;
    offset += multiplier * rowSize * rowIdx;
    multiplier *= rowSize;
    offset += multiplier * channelSize * channelIdx;
    multiplier *= channelSize;
    offset += multiplier * batchSize * batchIdx;

    return offset;
}

Tensor AllocateTensor(const TensorInfo& info)
{
    const auto byteSize = info.ByteSize();
    void* dataPtr = (void*)malloc(byteSize);
    return Tensor(dataPtr, info);
}

void CopyTensor(Tensor& source, Tensor& destination)
{
    assert(source.Info == destination.Info);
    assert(source.Info.ByteSize() == destination.Info.ByteSize());
    const size_t ByteSize = source.Info.ByteSize();

    std::memcpy(destination.DataPtr, source.DataPtr, ByteSize);
}
} // namespace CubbyDNN
