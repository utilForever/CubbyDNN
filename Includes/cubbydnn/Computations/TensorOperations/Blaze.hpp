/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifdef WITH_BLAZE
#ifndef CUBBYDNN_BLAZEOPS_HPP
#define CUBBYDNN_BLAZEOPS_HPP

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

class Blaze
{
public:
    template <typename T, bool IsAligned = false>
    static void TensorMul(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output)
    {
        const auto inputShapeA = inputA.TensorShape;
        const auto inputShapeB = inputB.TensorShape;
        const auto outputShape = output.TensorShape;

        const auto batchSizeA = inputShapeA.BatchSize();
        const auto batchSizeB = inputShapeB.BatchSize();
        const auto batchOutputSize = outputShape.BatchSize();

        if (batchSizeA != batchSizeB || batchSizeA != batchOutputSize)
            throw std::runtime_error("TensorMul - batch size mismatch");

        if (inputShapeA.BatchSize() != inputShapeB.BatchSize() ||
            inputShapeA.BatchSize() != outputShape.BatchSize())
            throw std::runtime_error("TensorMul - batch size mismatch");

        const auto colDataSizeA =
            inputA.PadSize > 0 ? inputA.PadSize : inputA.TensorShape.NumCols();
        const auto colDataSizeB =
            inputB.PadSize > 0 ? inputB.PadSize : inputB.TensorShape.NumCols();
        const auto colDataSizeOutput =
            output.PadSize > 0 ? output.PadSize : output.TensorShape.NumCols();

        const auto matrixSizeA = inputShapeA.NumRows() * colDataSizeA;
        const auto matrixSizeB = inputShapeB.NumRows() * colDataSizeB;
        const auto matrixSizeOut = outputShape.NumRows() * colDataSizeOutput;

        if constexpr (IsAligned)
            for (std::size_t batchIdx = 0; batchIdx < batchSizeA; ++batchIdx)
            {
                const CustomMatrix<T, aligned, padded, blaze::rowMajor> A(
                    static_cast<T*>(inputA.DataPtr) + batchIdx * matrixSizeA,
                    inputShapeA.NumRows(), inputShapeA.NumCols(),
                    inputA.PadSize);

                const CustomMatrix<T, aligned, padded, blaze::rowMajor> B(
                    static_cast<T*>(inputB.DataPtr) + batchIdx * matrixSizeB,
                    inputShapeB.NumRows(), inputShapeB.NumCols(),
                    inputB.PadSize);

                CustomMatrix<T, aligned, padded, blaze::rowMajor> Out(
                    static_cast<T*>(output.DataPtr) + batchIdx * matrixSizeOut,
                    outputShape.NumRows(), outputShape.NumCols(),
                    output.PadSize);

                Out = A * B;
            }
        else
        {
            for (std::size_t batchIdx = 0; batchIdx < batchSizeA; ++batchIdx)
            {
                const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                    static_cast<T*>(inputA.DataPtr) + batchIdx * matrixSizeA,
                    inputShapeA.NumRows(), inputShapeA.NumCols());

                const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> B(
                    static_cast<T*>(inputB.DataPtr) + batchIdx * matrixSizeB,
                    inputShapeB.NumRows(), inputShapeB.NumCols());

                CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> Out(
                    static_cast<T*>(output.DataPtr) + batchIdx * matrixSizeOut,
                    outputShape.NumRows(), outputShape.NumCols());

                Out = A * B;
            }
        }
    }

    template <typename T, bool IsAligned = false>
    static void TensorAdd(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output)
    {
        const auto inputShapeA = inputA.TensorShape;
        const auto inputShapeB = inputB.TensorShape;
        const auto outputShape = output.TensorShape;

        if (inputShapeA.BatchSize() != inputShapeB.BatchSize() ||
            inputShapeA.BatchSize() != outputShape.BatchSize())
            throw std::runtime_error("TensorAdd - batch size mismatch");

        const auto batchSize = inputShapeA.BatchSize();

        const auto colDataSizeA =
            inputA.PadSize > 0 ? inputA.PadSize : inputA.TensorShape.NumCols();
        const auto colDataSizeB =
            inputB.PadSize > 0 ? inputB.PadSize : inputB.TensorShape.NumCols();
        const auto colDataSizeOutput =
            output.PadSize > 0 ? output.PadSize : output.TensorShape.NumCols();

        const auto matrixSizeA = inputShapeA.NumRows() * colDataSizeA;
        const auto matrixSizeB = inputShapeB.NumRows() * colDataSizeB;
        const auto matrixSizeOut = outputShape.NumRows() * colDataSizeOutput;

        if constexpr (IsAligned)
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const CustomMatrix<T, aligned, padded, blaze::rowMajor> A(
                    static_cast<T*>(inputA.DataPtr) + batchIdx * matrixSizeA,
                    inputShapeA.NumRows(), inputShapeB.NumCols(),
                    inputA.PadSize);

                const CustomMatrix<T, aligned, padded, blaze::rowMajor> B(
                    static_cast<T*>(inputB.DataPtr) + batchIdx * matrixSizeB,
                    inputShapeB.NumRows(), inputShapeB.NumCols(),
                    inputB.PadSize);

                CustomMatrix<T, aligned, padded, blaze::rowMajor> Out(
                    static_cast<T*>(output.DataPtr) + batchIdx * matrixSizeOut,
                    outputShape.NumRows(), outputShape.NumCols(),
                    output.PadSize);

                Out = A + B;
            }
        else
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                    static_cast<T*>(inputA.DataPtr) + batchIdx * matrixSizeA,
                    inputShapeA.NumRows(), inputShapeB.NumCols());

                const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> B(
                    static_cast<T*>(inputB.DataPtr) + batchIdx * matrixSizeB,
                    inputShapeB.NumRows(), inputShapeB.NumCols());

                CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> Out(
                    static_cast<T*>(output.DataPtr) + batchIdx * matrixSizeOut,
                    outputShape.NumRows(), outputShape.NumCols());

                Out = A + B;
            }
    }

    template <typename T, bool IsAligned = false>
    static void TensorTranspose(const Tensor& input, Tensor& output)
    {
        const auto inputShape = input.TensorShape;
        const auto outputShape = output.TensorShape;

        const auto colDataSizeInput =
            input.PadSize > 0 ? input.PadSize : input.TensorShape.NumCols();
        const auto colDataSizeOutput =
            output.PadSize > 0 ? output.PadSize : output.TensorShape.NumCols();

        const auto matrixSizeInput = inputShape.NumRows() * colDataSizeInput;
        const auto matrixSizeOutput = outputShape.NumRows() * colDataSizeOutput;

        if (inputShape.BatchSize() != outputShape.BatchSize())
            throw std::runtime_error("TensorTranspose - Batch size mismatch");

        const auto batchSize = inputShape.BatchSize();

        if (IsAligned)
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const CustomMatrix<T, aligned, padded, blaze::rowMajor> A(
                    static_cast<T*>(input.DataPtr) +
                    batchIdx * matrixSizeInput,
                    inputShape.NumRows(), inputShape.NumCols(),
                    input.PadSize);

                CustomMatrix<T, aligned, padded, blaze::rowMajor> Out(
                    static_cast<T*>(output.DataPtr) +
                    batchIdx * matrixSizeOutput,
                    outputShape.NumRows(), outputShape.NumCols(),
                    output.PadSize);

                Out = blaze::trans(A);
            }
        else
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                    static_cast<T*>(input.DataPtr) +
                    batchIdx * matrixSizeInput,
                    inputShape.NumRows(), inputShape.NumCols()
                    );

                CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> Out(
                    static_cast<T*>(output.DataPtr) +
                    batchIdx * matrixSizeOutput,
                    outputShape.NumRows(), outputShape.NumCols());

                Out = blaze::trans(A);
            }
    }
};
} // namespace CubbyDNN

#endif
#endif
