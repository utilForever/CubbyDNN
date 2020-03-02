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
using blaze::unaligned;
using blaze::unpadded;

template <typename T>
void GetIdentityMatrix(Tensor& tensor)
{
    const auto batchSize = tensor.Info.GetShape().Batch;
    const auto channelSize = tensor.Info.GetShape().Channel;
    const auto rowSize = tensor.Info.GetShape().Row;

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t channelIdx = 0; channelIdx < channelSize; ++channelIdx)
            for (std::size_t idx = 0; idx < rowSize; ++idx)
            {
                const auto offset =
                    tensor.GetElementOffset({ batchIdx, idx, channelIdx, idx });
                *(static_cast<T*>(tensor.DataPtr) + offset) =
                    static_cast<T>(1.0f);
            }
}

template <typename T>
void TensorMul(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    const auto inputShapeA = inputA.Info.GetShape();
    const auto inputShapeB = inputB.Info.GetShape();
    const auto outputShape = output.Info.GetShape();

    assert(inputShapeA.Batch == inputShapeB.Batch &&
           inputShapeA.Batch == outputShape.Batch);
    assert(inputShapeA.Channel == inputShapeB.Channel &&
           inputShapeA.Channel == outputShape.Channel);
    assert(inputShapeA.Col == inputShapeB.Row);

    const auto channelSize = inputShapeA.Channel;
    const auto batchSize = inputShapeA.Batch;

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t channelIdx = 0; channelIdx < channelSize; ++channelIdx)
        {
            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                static_cast<T*>(inputA.DataPtr) +
                    output.GetElementOffset({ batchIdx, channelIdx, 0, 0 }),
                inputShapeA.Row, inputShapeB.Col);

            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> B(
                static_cast<T*>(inputB.DataPtr) +
                    output.GetElementOffset({ batchIdx, channelIdx, 0, 0 }),
                inputShapeB.Row, inputShapeB.Col);

            CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> Out(
                static_cast<T*>(output.DataPtr) +
                    output.GetElementOffset({ batchIdx, channelIdx, 0, 0 }),
                outputShape.Row, outputShape.Col);

            Out = A * B;
        }
}

template <typename T>
void TensorAdd(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    const auto inputShapeA = inputA.Info.GetShape();
    const auto inputShapeB = inputB.Info.GetShape();
    const auto outputShape = output.Info.GetShape();

    assert(inputShapeA.Row == inputShapeB.Row &&
           inputShapeA.Row == outputShape.Row);
    assert(inputShapeA.Col == inputShapeB.Col &&
           inputShapeA.Col == outputShape.Col);
    assert(inputShapeA.Row == inputShapeB.Row &&
           inputShapeA.Batch == outputShape.Batch);
    assert(inputShapeA.Channel == inputShapeB.Channel &&
           inputShapeA.Channel == outputShape.Channel);

    const auto channelSize = inputShapeA.Channel;
    const auto batchSize = inputShapeA.Batch;

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t channelIdx = 0; channelIdx < channelSize; ++channelIdx)
        {
            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                static_cast<T*>(inputA.DataPtr) +
                    output.GetElementOffset({ batchIdx, channelIdx, 0, 0 }),
                inputShapeA.Row, inputShapeB.Col);

            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> B(
                static_cast<T*>(inputB.DataPtr) +
                    output.GetElementOffset({ batchIdx, channelIdx, 0, 0 }),
                inputShapeB.Row, inputShapeB.Col);

            CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> Out(
                static_cast<T*>(output.DataPtr) +
                    output.GetElementOffset({ batchIdx, channelIdx, 0, 0 }),
                outputShape.Row, outputShape.Col);

            Out = A + B;
        }
}

template <typename T>
void TensorTranspose(const Tensor& input, Tensor& output)
{
    const auto inputShape = input.Info.GetShape();
    const auto outputShape = output.Info.GetShape();

    assert(inputShape.Batch == outputShape.Batch);
    assert(inputShape.Channel == outputShape.Channel);
    assert(inputShape.Row == outputShape.Col);
    assert(inputShape.Col == outputShape.Row);

    const auto channelSize = inputShape.Channel;
    const auto batchSize = inputShape.Batch;

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t channelIdx = 0; channelIdx < channelSize; ++channelIdx)
        {
            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                static_cast<T*>(input.DataPtr) +
                    output.GetElementOffset({ batchIdx, channelIdx, 0, 0 }),
                inputShape.Row, inputShape.Col);

            CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> Out(
                static_cast<T*>(output.DataPtr) +
                    output.GetElementOffset({ batchIdx, channelIdx, 0, 0 }),
                outputShape.Row, outputShape.Col);

            Out = blaze::trans(A);
        }
}

template <typename T>
void TensorReshape(const Tensor& input, Tensor& output)
{

}
}  // namespace CubbyDNN

#endif
