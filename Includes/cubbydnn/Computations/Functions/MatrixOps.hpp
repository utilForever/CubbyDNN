/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_MATRIX_HPP
#define CUBBYDNN_MATRIX_HPP

#include <blaze/math/CustomMatrix.h>
#include <cubbydnn/Tensors/Tensor.hpp>

namespace CubbyDNN
{
using blaze::aligned;
using blaze::columnMajor;
using blaze::CustomMatrix;
using blaze::padded;
using blaze::rowMajor;
using blaze::unaligned;
using blaze::unpadded;


template<typename T>
static void GetIdentityMatrix(const Tensor& tensor)
{

    const auto batchSize = tensor.Info.GetShape().BatchSize;
    const auto channelSize = tensor.Info.GetShape().ChannelSize;
    const auto rowSize = tensor.Info.GetShape().RowSize;
    const auto colSize = tensor.Info.GetShape().ColSize;
    assert(rowSize == colSize);

    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (size_t channelIdx = 0; channelIdx < channelSize; ++channelIdx)
            for (size_t idx = 0; idx < rowSize; ++idx)
            {
                const auto offset = tensor.GetElementOffset({ batchIdx, idx, channelIdx, idx });
                *(static_cast<T*>(tensor.DataPtr) + offset) = static_cast<T>(1.0f);
            }
}

template<typename T>
static void MatMul(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    const auto inputShapeA = inputA.Info.GetShape();
    const auto inputShapeB = inputA.Info.GetShape();
    const auto outputShape = output.Info.GetShape();

    assert(inputShapeA.BatchSize == inputShapeB.BatchSize &&
        inputShapeA.BatchSize == outputShape.BatchSize);
    assert(inputShapeA.ChannelSize == inputShapeB.ChannelSize &&
        inputShapeA.ChannelSize == outputShape.ChannelSize);
    assert(inputShapeA.ColSize == inputShapeB.RowSize);

    const auto channelSize = inputShapeA.ChannelSize;
    const auto batchSize = inputShapeA.BatchSize;
    const auto loopSize = channelSize * batchSize;

    for (size_t batchIdx = 0; batchIdx < loopSize; ++batchIdx)
        for (size_t channelIdx = 0; channelIdx < channelSize; ++channelIdx)
        {
            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                static_cast<T*>(inputA.DataPtr) +
                output.GetElementOffset({ batchIdx, 0, channelIdx, 0 }),
                inputShapeA.RowSize, inputShapeB.ColSize);

            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> B(
                static_cast<T*>(inputB.DataPtr) +
                output.GetElementOffset({ batchIdx, 0, channelIdx, 0 }),
                inputShapeB.RowSize, inputShapeB.ColSize);

            CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> C(
                static_cast<T*>(output.DataPtr) +
                output.GetElementOffset({ batchIdx, 0, channelIdx, 0 }),
                outputShape.RowSize, outputShape.ColSize);

            C = A * B;
        }
}
} // namespace CubbyDNN

#endif
