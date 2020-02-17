/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_MATRIX_HPP
#define CUBBYDNN_MATRIX_HPP

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)
#pragma warning(disable : 26812)
#pragma warning(disable : 6294)
#pragma warning(disable : 6993)
#endif
#include <blaze/math/CustomMatrix.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
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

static void MatMul(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    const auto inputShapeA = inputA.Info.GetShape();
    const auto inputShapeB = inputA.Info.GetShape();
    const auto outputShape = output.Info.GetShape();

    const auto shapeAIdx = inputA.Info.GetShapeIndex();
    const auto shapeBIdx = inputB.Info.GetShapeIndex();
    const auto shapeCIdx = output.Info.GetShapeIndex();

    assert(shapeAIdx.BatchSizeIdx == shapeBIdx.BatchSizeIdx &&
           shapeAIdx.BatchSizeIdx == shapeCIdx.BatchSizeIdx);
    assert(shapeAIdx.ChannelSizeIdx == shapeBIdx.ChannelSizeIdx &&
           shapeAIdx.ChannelSizeIdx == shapeCIdx.ChannelSizeIdx);
    assert(shapeAIdx.ColSizeIdx == shapeBIdx.RowSizeIdx);

    const auto channelSize = inputShapeA.at(shapeAIdx.ChannelSizeIdx);
    const auto batchSize = inputShapeA.at(shapeAIdx.BatchSizeIdx);
    const auto loopSize = channelSize * batchSize;

    for (size_t batchIdx = 0; batchIdx < loopSize; ++batchIdx)
        for (size_t channelIdx = 0; channelIdx < channelSize; ++channelIdx)
        {
            const CustomMatrix<float, unaligned, unpadded, blaze::rowMajor> A(
                static_cast<float*>(inputA.DataPtr) +
                    output.GetElementOffset({ batchIdx, 0, channelIdx, 0 }),
                inputShapeA.at(shapeAIdx.RowSizeIdx),
                inputShapeB.at(shapeAIdx.ColSizeIdx));

            const CustomMatrix<float, unaligned, unpadded, blaze::rowMajor> B(
                static_cast<float*>(inputB.DataPtr) +
                    output.GetElementOffset({ batchIdx, 0, channelIdx, 0 }),
                inputShapeB.at(shapeBIdx.RowSizeIdx),
                inputShapeB.at(shapeBIdx.ColSizeIdx));

            CustomMatrix<float, unaligned, unpadded, blaze::rowMajor> C(
                static_cast<float*>(output.DataPtr) +
                    output.GetElementOffset({ batchIdx, 0, channelIdx, 0 }),
                outputShape.at(shapeCIdx.RowSizeIdx),
                outputShape.at(shapeCIdx.ColSizeIdx));

            C = A * B;
        }
}

} // namespace CubbyDNN

#endif
