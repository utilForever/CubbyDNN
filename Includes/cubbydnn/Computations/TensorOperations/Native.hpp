// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTETENSOR_HPP
#define CUBBYDNN_COMPUTETENSOR_HPP

#include <cubbydnn/Tensors/Tensor.hpp>
#include <functional>

namespace CubbyDNN::Compute
{
class Native
{
public:
    template <typename T>
    static void BasicLoop(const Tensor& input, Tensor& output,
                          const std::function<T(T&)>& function)
    {
        const auto inputShape = input.TensorShape;
        const auto outputShape = output.TensorShape;

        const auto batchSize = inputShape.BatchSize();

        const auto numRows = inputShape.NumRows();
        const auto numCols = inputShape.NumCols();

        for (std::size_t batch = 0; batch < batchSize; ++batch)
            for (std::size_t i = 0; i < numRows; ++i)
                for (std::size_t j = 0; j < numCols; ++j)
                    static_cast<T*>(output.DataPtr.get())[i * numCols + j] = function(
                        static_cast<T*>(input.DataPtr.get())[i * numCols + j]);
    }

    template <typename T>
    static void TensorAdd(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output)
    {
        const auto inputShapeA = inputA.TensorShape;
        const auto inputShapeB = inputB.TensorShape;
        const auto outputShape = output.TensorShape;

        const auto numRows = inputShapeA.NumRows();
        const auto numCols = inputShapeA.NumCols();
        const auto batchSize = inputShapeA.BatchSize();
        const auto matrixSize = numRows * numCols;

        const T* inputPtrA = static_cast<T*>(inputA.DataPtr.get());
        const T* inputPtrB = static_cast<T*>(inputB.DataPtr.get());
        T* outputPtr = static_cast<T*>(output.DataPtr.get());

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t i = 0; i < numRows; ++i)
                for (std::size_t j = 0; j < numCols; ++j)
                {
                    outputPtr[batchIdx * matrixSize + i * numCols + j] =
                        inputPtrA[batchIdx * matrixSize + i * numCols + j] +
                        inputPtrB[batchIdx * matrixSize + i * numCols + j];
                }
    }

    template <typename T>
    static void TensorAdd(Tensor& tensor, const Tensor& toAdd)
    {
        const auto tensorShape = tensor.TensorShape;
        const auto addShape = toAdd.TensorShape;

        const auto numRows = tensorShape.NumRows();
        const auto numCols = tensorShape.NumCols();
        const auto batchSize = tensorShape.BatchSize();
        const auto matrixSize = numRows * numCols;

        T* tensorPtr = static_cast<T*>(tensor.DataPtr.get());
        const T* addPtr = static_cast<T*>(toAdd.DataPtr.get());

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t i = 0; i < numRows; ++i)
                for (std::size_t j = 0; j < numCols; ++j)
                {
                    tensorPtr[batchIdx * matrixSize + i * numCols + j] =
                        addPtr[batchIdx * matrixSize + i * numCols + j];
                }
    }

    template <typename T, std::size_t blockSize = 32>
    static void TensorMul(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output)
    {
        const auto inputShapeA = inputA.TensorShape;
        const auto inputShapeB = inputB.TensorShape;
        const auto outputShape = output.TensorShape;

        const auto batchSize = inputShapeA.BatchSize();
        const auto matrixSizeA = inputShapeA.NumRows() * inputShapeA.NumCols();
        const auto matrixSizeB = inputShapeB.NumRows() * inputShapeB.NumCols();
        const auto matrixSizeOutput =
            outputShape.NumRows() * outputShape.NumCols();

        const auto numRowsA = inputShapeA.NumRows();
        const auto numColsA = inputShapeA.NumCols();

        const auto numRowsB = inputShapeB.NumRows();
        const auto numColsB = inputShapeB.NumCols();

        const T* inputAPtr = static_cast<T*>(inputA.DataPtr.get());
        const T* inputBPtr = static_cast<T*>(inputB.DataPtr.get());
        T* outputPtr = static_cast<T*>(output.DataPtr.get());

        //! cache friendly matrix multiplication minimizing cache misses
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t ii = 0; ii < numRowsA; ii += blockSize)
                for (std::size_t jj = 0; jj < numColsB; jj += blockSize)
                    for (std::size_t kk = 0; kk < numRowsB; kk += blockSize)
                    {
                        std::size_t i_lim = ii + blockSize;
                        if (i_lim > numRowsA)
                            i_lim = numRowsA;

                        std::size_t j_lim = jj + blockSize;
                        if (j_lim > numColsB)
                            j_lim = numColsB;

                        std::size_t k_lim = kk + blockSize;
                        if (k_lim > numRowsB)
                            k_lim = numRowsB;

                        for (std::size_t i = ii; i < i_lim; i++)
                            for (std::size_t j = jj; j < j_lim; j++)
                                for (std::size_t k = kk; k < k_lim; k++)
                                    outputPtr[
                                            batchIdx * matrixSizeOutput + i *
                                            numColsB + j] +=
                                        inputAPtr[
                                            batchIdx * matrixSizeA + i *
                                            numColsA + k] *
                                        inputBPtr[
                                            batchIdx * matrixSizeB + k *
                                            numColsB + j];
                    }
    }

    template <typename T, std::size_t blockSize = 32>
    static void TensorTranspose(const Tensor& input, Tensor& output)
    {
        const auto shape = input.TensorShape;

        const auto numRows = shape.NumRows();
        const auto numCols = shape.NumCols();

        const auto matrixSize = numRows * numCols;
        const auto batchSize = shape.BatchSize();

        const T* inputPtr = static_cast<T*>(input.DataPtr.get());
        T* outputPtr = static_cast<T*>(output.DataPtr.get());

        //! Optimized matrix transpose minimizing cache misses
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (std::size_t ii = 0; ii < numRows; ii += blockSize)
                for (std::size_t jj = 0; jj < numCols; jj += blockSize)
                {
                    std::size_t i_lim = ii + blockSize;
                    if (i_lim > numRows)
                        i_lim = numRows;

                    std::size_t j_lim = jj + blockSize;
                    if (j_lim > numCols)
                        j_lim = numCols;

                    for (std::size_t i = ii; i < i_lim; i++)
                        for (std::size_t j = jj; j < j_lim; j++)
                            outputPtr[batchIdx * matrixSize + j * numRows + i] =
                                inputPtr[batchIdx * matrixSize + i * numCols + j
                                ];
                }
        }
    }

    template <typename T>
    static void TensorDot(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output)
    {
        const auto shape = inputA.TensorShape;

        const auto numRows = shape.NumRows();
        const auto numCols = inputA.GetPaddedNumCols();

        const auto matrixSize = numRows * numCols;
        const auto batchSize = shape.BatchSize();

        const T* inputPtrA = static_cast<T*>(inputA.DataPtr.get());
        const T* inputPtrB = static_cast<T*>(inputB.DataPtr.get());
        T* outputPtr = static_cast<T*>(output.DataPtr.get());

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t i = 0; i < numRows; ++i)
                for (std::size_t j = 0; j < numCols; ++j)
                {
                    outputPtr[batchIdx * matrixSize + i * numCols + j] =
                        inputPtrA[batchIdx * matrixSize + i * numCols + j] *
                        inputPtrB[batchIdx * matrixSize + i * numCols + j];
                }
    }
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTETENSOR_HPP
