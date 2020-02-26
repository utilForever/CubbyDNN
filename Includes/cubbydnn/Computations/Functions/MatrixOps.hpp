/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_MATRIXOP_HPP
#define CUBBYDNN_MATRIXOP_HPP

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
void GetIdentityMatrix(Tensor& tensor)
{

    const auto batchSize = tensor.Info.GetShape().Batch;
    const auto channelSize = tensor.Info.GetShape().Channel;
    const auto rowSize = tensor.Info.GetShape().Row;

    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (size_t channelIdx = 0; channelIdx < channelSize; ++channelIdx)
            for (size_t idx = 0; idx < rowSize; ++idx)
            {
                const auto offset = tensor.GetElementOffset({ batchIdx, idx, channelIdx, idx });
                *(static_cast<T*>(tensor.DataPtr) + offset) = static_cast<T>(1.0f);
            }
}

template<typename T>
void MatMul(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    const auto inputShapeA = inputA.Info.GetShape();
    const auto inputShapeB = inputA.Info.GetShape();
    const auto outputShape = output.Info.GetShape();

    assert(inputShapeA.Batch == inputShapeB.Batch &&
        inputShapeA.Batch == outputShape.Batch);
    assert(inputShapeA.Channel == inputShapeB.Channel &&
        inputShapeA.Channel == outputShape.Channel);
    assert(inputShapeA.Col == inputShapeB.Row);

    const auto channelSize = inputShapeA.Channel;
    const auto batchSize = inputShapeA.Batch;

    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (size_t channelIdx = 0; channelIdx < channelSize; ++channelIdx)
        {
            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                static_cast<T*>(inputA.DataPtr) +
                output.GetElementOffset({ batchIdx, channelIdx, 0, 0 }),
                inputShapeA.Row, inputShapeB.Col);

            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> B(
                static_cast<T*>(inputB.DataPtr) +
                output.GetElementOffset({ batchIdx, channelIdx, 0, 0 }),
                inputShapeB.Row, inputShapeB.Col);

            CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> C(
                static_cast<T*>(output.DataPtr) +
                output.GetElementOffset({ batchIdx, channelIdx, 0, 0 }),
                outputShape.Row, outputShape.Col);

            C = A * B;
        }
}
} // namespace CubbyDNN

#endif
