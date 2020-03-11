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
using blaze::columnMajor;
using blaze::CustomMatrix;
using blaze::aligned;
using blaze::padded;
using blaze::unaligned;
using blaze::unpadded;


template <typename T, bool IsAligned = false>
void TensorMul(const Tensor& inputA, const Tensor& inputB,
               Tensor& output)
{
    const auto inputShapeA = inputA.Info.GetShape();
    const auto inputShapeB = inputB.Info.GetShape();
    const auto outputShape = output.Info.GetShape();

    if (inputShapeA.BatchSize() != inputShapeB.BatchSize() ||
        inputShapeA.BatchSize() != outputShape.BatchSize())
        throw std::runtime_error("TensorMul - batch size mismatch");

    const auto batchSize = inputShapeA.BatchSize();

    const auto matrixSizeA = inputShapeA.PaddedMatrixSize();
    const auto matrixSizeB = inputShapeB.PaddedMatrixSize();
    const auto matrixSizeOut = outputShape.PaddedMatrixSize();

    if constexpr (IsAligned)
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const CustomMatrix<T, aligned, padded, blaze::rowMajor> A(
                static_cast<T*>(inputA.DataPtr) + batchIdx * matrixSizeA,
                inputShapeA.Row(), inputShapeA.Col(),
                inputShapeA.PadSize());

            const CustomMatrix<T, aligned, padded, blaze::rowMajor> B(
                static_cast<T*>(inputB.DataPtr) + batchIdx * matrixSizeB,
                inputShapeB.Row(), inputShapeB.Col(),
                inputShapeB.PadSize());

            CustomMatrix<T, aligned, padded, blaze::rowMajor> Out(
                static_cast<T*>(output.DataPtr) + batchIdx * matrixSizeOut,
                outputShape.Row(), outputShape.Col(),
                outputShape.PadSize());

            Out = A * B;
        }
    else
    {
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                static_cast<T*>(inputA.DataPtr) + batchIdx * matrixSizeA,
                inputShapeA.Row(), inputShapeA.Col());

            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> B(
                static_cast<T*>(inputB.DataPtr) + batchIdx * matrixSizeB,
                inputShapeB.Row(), inputShapeB.Col());

            CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> Out(
                static_cast<T*>(output.DataPtr) + batchIdx * matrixSizeOut,
                outputShape.Row(), outputShape.Col());

            Out = A * B;
        }
    }
}

template <typename T, bool IsAligned = false>
void TensorAdd(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    const auto inputShapeA = inputA.Info.GetShape();
    const auto inputShapeB = inputB.Info.GetShape();
    const auto outputShape = output.Info.GetShape();

    if (inputShapeA.BatchSize() != inputShapeB.BatchSize() ||
        inputShapeA.BatchSize() != outputShape.BatchSize())
        throw std::runtime_error("TensorAdd - batch size mismatch");

    const auto batchSize = inputShapeA.BatchSize();

    const auto matrixSizeA = inputShapeA.PaddedMatrixSize();
    const auto matrixSizeB = inputShapeB.PaddedMatrixSize();
    const auto matrixSizeOut = outputShape.PaddedMatrixSize();

    if constexpr (IsAligned)
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const CustomMatrix<T, aligned, padded, blaze::rowMajor> A(
                static_cast<T*>(inputA.DataPtr) + batchIdx * matrixSizeA,
                inputShapeA.Row(), inputShapeB.Col(),
                inputShapeA.PadSize());

            const CustomMatrix<T, aligned, padded, blaze::rowMajor> B(
                static_cast<T*>(inputB.DataPtr) + batchIdx * matrixSizeB,
                inputShapeB.Row(), inputShapeB.Col(),
                inputShapeB.PadSize());

            CustomMatrix<T, aligned, padded, blaze::rowMajor> Out(
                static_cast<T*>(output.DataPtr) + batchIdx * matrixSizeOut,
                outputShape.Row(), outputShape.Col(),
                outputShape.PadSize());

            Out = A + B;
        }
    else
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                static_cast<T*>(inputA.DataPtr) + batchIdx * matrixSizeA,
                inputShapeA.Row(), inputShapeB.Col());

            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> B(
                static_cast<T*>(inputB.DataPtr) + batchIdx * matrixSizeB,
                inputShapeB.Row(), inputShapeB.Col());

            CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> Out(
                static_cast<T*>(output.DataPtr) + batchIdx * matrixSizeOut,
                outputShape.Row(), outputShape.Col());

            Out = A + B;
        }
}

template <typename T, bool IsAligned = false>
void TensorTranspose(const Tensor& input, Tensor& output)
{
    const auto inputShape = input.Info.GetShape();
    const auto outputShape = output.Info.GetShape();

    if (inputShape.BatchSize() != outputShape.BatchSize())
        throw std::runtime_error("TensorTranspose - Batch size mismatch");

    const auto batchSize = inputShape.BatchSize();

    if (IsAligned)
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const CustomMatrix<T, aligned, padded, blaze::rowMajor> A(
                static_cast<T*>(input.DataPtr) +
                batchIdx * inputShape.PaddedMatrixSize(),
                inputShape.Row(), inputShape.Col(),
                inputShape.PadSize());

            CustomMatrix<T, aligned, padded, blaze::rowMajor> Out(
                static_cast<T*>(output.DataPtr) +
                batchIdx * outputShape.PaddedMatrixSize(),
                outputShape.Row(), outputShape.Col(),
                outputShape.PadSize());

            Out = blaze::trans(A);
        }
    else
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                static_cast<T*>(input.DataPtr) +
                batchIdx * inputShape.PaddedMatrixSize(),
                inputShape.Row(), inputShape.Col(),
                inputShape.PadSize());

            CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> Out(
                static_cast<T*>(output.DataPtr) +
                batchIdx * outputShape.PaddedMatrixSize(),
                outputShape.Row(), outputShape.Col(),
                outputShape.PadSize());

            Out = blaze::trans(A);
        }
}
} // namespace CubbyDNN

#endif
