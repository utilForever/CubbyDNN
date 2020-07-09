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

        const std::size_t colDataSizeInput = input.GetColumnElementSize();
        const std::size_t colDataSizeOutput = output.GetColumnElementSize();

        const auto batchSize = inputShape.NumMatrices();

        const auto numRows = inputShape.NumRows();
        const auto numCols = inputShape.NumCols();

        for (std::size_t batch = 0; batch < batchSize; ++batch)
            for (std::size_t i = 0; i < numRows; ++i)
                for (std::size_t j = 0; j < numCols; ++j)
                    static_cast<T*>(
                            output.DataPtr)[i * colDataSizeOutput + j] =
                        function(static_cast<T*>(
                            input.DataPtr)[i * colDataSizeInput + j]);
    }

    template <typename T>
    static void TensorAdd(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output)
    {
        const auto inputShapeA = inputA.TensorShape;
        const auto inputShapeB = inputB.TensorShape;
        const auto outputShape = output.TensorShape;

        const std::size_t colDataSizeA = inputA.GetColumnElementSize();
        const std::size_t colDataSizeB = inputB.GetColumnElementSize();
        const std::size_t colDataSizeOutput = output.GetColumnElementSize();

        const auto numRows = inputShapeA.NumRows();
        const auto numCols = inputShapeA.NumCols();
        const auto batchSize = inputShapeA.NumMatrices();
        const auto matrixSizeA = numRows * colDataSizeA;
        const auto matrixSizeB = numRows * colDataSizeB;
        const auto matrixSizeOutput = numRows * colDataSizeOutput;

        const T* inputPtrA = static_cast<T*>(inputA.DataPtr);
        const T* inputPtrB = static_cast<T*>(inputB.DataPtr);
        T* outputPtr = static_cast<T*>(output.DataPtr);

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t i = 0; i < numRows; ++i)
                for (std::size_t j = 0; j < numCols; ++j)
                {
                    outputPtr[batchIdx * matrixSizeOutput +
                              i * colDataSizeOutput + j] =
                        inputPtrA[batchIdx * matrixSizeA + i * colDataSizeA +
                                  j] +
                        inputPtrB[batchIdx * matrixSizeB + i * colDataSizeB +
                                  j];
                }
    }

    template <typename T>
    static void TensorAdd(Tensor& tensor, const Tensor& toAdd)
    {
        const auto tensorShape = tensor.TensorShape;
        const auto addShape = toAdd.TensorShape;

        const std::size_t colDataSizeInput = tensor.GetColumnElementSize();
        const std::size_t colDataSizeAdd = toAdd.GetColumnElementSize();

        const auto numRows = tensorShape.NumRows();
        const auto numCols = tensorShape.NumCols();
        const auto batchSize = tensorShape.NumMatrices();
        const auto matrixSizeTensor = numRows * colDataSizeInput;
        const auto matrixSizeAdd = numRows * colDataSizeAdd;

        T* tensorPtr = static_cast<T*>(tensor.DataPtr);
        const T* addPtr = static_cast<T*>(toAdd.DataPtr);

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t i = 0; i < numRows; ++i)
                for (std::size_t j = 0; j < numCols; ++j)
                {
                    tensorPtr[batchIdx * matrixSizeTensor +
                              i * colDataSizeInput + j] +=
                        addPtr[batchIdx * matrixSizeAdd + i * colDataSizeAdd +
                               j];
                }
    }

    template <typename T>
    static void BatchMean(const Tensor& input, std::size_t idx, Tensor& output)
    {
        std::size_t interval = 1;
        for (auto i = input.TensorShape.Dim() - 1; i > idx; --i)
            if (i == input.TensorShape.Dim() - 1)
                interval *= input.GetColumnElementSize();
            else
                interval *= input.TensorShape.At(i);

        std::size_t batchSize = 1;
        for (std::size_t i = 0; i <= idx; ++i)
            if (i == input.TensorShape.Dim() - 1)
                batchSize *= input.GetColumnElementSize();
            else
                batchSize *= input.TensorShape.At(i);

        const T* tensorPtr = static_cast<T*>(input.DataPtr);
        T* outputPtr = static_cast<T*>(output.DataPtr);

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (std::size_t elementIdx = 0; elementIdx < interval;
                 ++elementIdx)
                outputPtr[elementIdx] +=
                    tensorPtr[batchIdx * interval + elementIdx];
        }

        for (std::size_t elementIdx = 0; elementIdx < interval; ++elementIdx)
            outputPtr[elementIdx] /= static_cast<T>(batchSize);
    }

    // TODO : Fix this to in-place transpose
    template <typename T, std::size_t blockSize = 32>
    static void TensorMul(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output, bool transposeA, bool transposeB)
    {
        const auto inputShapeA = inputA.TensorShape;
        const auto inputShapeB = inputB.TensorShape;
        const auto outputShape = output.TensorShape;

        const std::size_t colDataSizeA = inputA.GetColumnElementSize();
        const std::size_t colDataSizeB = inputB.GetColumnElementSize();
        const std::size_t colDataSizeOutput = output.GetColumnElementSize();

        const auto batchSize = inputShapeA.NumMatrices();
        const auto matrixSizeA = inputShapeA.NumRows() * colDataSizeA;
        const auto matrixSizeB = inputShapeB.NumRows() * colDataSizeB;
        const auto matrixSizeOutput = outputShape.NumRows() * colDataSizeOutput;

        const auto numRowsA = inputShapeA.NumRows();
        const auto numRowsB = inputShapeB.NumRows();
        const auto numColsB = inputShapeB.NumCols();

        const T* inputAPtr = static_cast<T*>(inputA.DataPtr);
        const T* inputBPtr = static_cast<T*>(inputB.DataPtr);
        T* outputPtr = static_cast<T*>(output.DataPtr);

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
                                {
                                    T valA =
                                        transposeA
                                            ? inputAPtr[batchIdx * matrixSizeA +
                                                        k * colDataSizeA + i]
                                            : inputAPtr[batchIdx * matrixSizeA +
                                                        i * colDataSizeA + k];

                                    T valB =
                                        transposeB
                                            ? inputBPtr[batchIdx * matrixSizeB +
                                                        j * colDataSizeB + k]
                                            : inputBPtr[batchIdx * matrixSizeB +
                                                        k * colDataSizeB + j];

                                    outputPtr[batchIdx * matrixSizeOutput +
                                              i * colDataSizeB + j] +=
                                        valA + valB;
                                }
                    }
    }

    template <typename T, std::size_t blockSize = 32>
    static void TensorTranspose(const Tensor& input, Tensor& output)
    {
        const auto shape = input.TensorShape;

        const std::size_t colDataSizeInput = input.GetColumnElementSize();
        const std::size_t colDataSizeOutput = output.GetColumnElementSize();

        const auto numRows = shape.NumRows();
        const auto numCols = shape.NumCols();

        const auto matrixSize = numRows * colDataSizeInput;
        const auto batchSize = shape.NumMatrices();

        const T* inputPtr = static_cast<T*>(input.DataPtr);
        T* outputPtr = static_cast<T*>(output.DataPtr);

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
                            outputPtr[batchIdx * matrixSize +
                                      j * colDataSizeOutput + i] =
                                inputPtr[batchIdx * matrixSize +
                                         i * colDataSizeInput + j];
                }
        }
    }

    template <typename T>
    static void TensorDot(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output)
    {
        const auto shape = inputA.TensorShape;

        const std::size_t colDataSizeA = inputA.GetColumnElementSize();
        const std::size_t colDataSizeB = inputB.GetColumnElementSize();
        const std::size_t colDataSizeOutput = output.GetColumnElementSize();

        const auto numRows = shape.NumRows();
        const auto numCols = shape.NumCols();

        const auto matrixSizeA = numRows * colDataSizeA;
        const auto matrixSizeB = numRows * colDataSizeB;
        const auto matrixSizeOutput = numRows * colDataSizeOutput;
        const auto batchSize = shape.NumMatrices();

        const T* inputPtrA = static_cast<T*>(inputA.DataPtr);
        const T* inputPtrB = static_cast<T*>(inputB.DataPtr);
        T* outputPtr = static_cast<T*>(output.DataPtr);

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t i = 0; i < numRows; ++i)
                for (std::size_t j = 0; j < numCols; ++j)
                {
                    outputPtr[batchIdx * matrixSizeOutput +
                              i * colDataSizeOutput + j] =
                        inputPtrA[batchIdx * matrixSizeA + i * colDataSizeA +
                                  j] *
                        inputPtrB[batchIdx * matrixSizeB + i * colDataSizeB +
                                  j];
                }
    }

    template <typename T>
    static void ScalarMul(Tensor& tensor, T toMul)
    {
        const auto shape = tensor.TensorShape;
        const auto totalSize = shape.NumMatrices() * shape.NumRows() *
                               tensor.GetColumnElementSize();

        T* dataPtr = static_cast<T*>(tensor.DataPtr);
        for (std::size_t i = 0; i < totalSize; ++i)
        {
            dataPtr[i] *= toMul;
        }
    }

    template <typename T>
    static void ScalarMul(const Tensor& input, Tensor& output, T toMul)
    {
        const auto inputShape = input.TensorShape;

        const std::size_t colDataSizeInput = input.GetColumnElementSize();
        const std::size_t colDataSizeOutput = output.GetColumnElementSize();

        const auto numRows = inputShape.NumRows();
        const auto numCols = inputShape.NumCols();
        const auto batchSize = inputShape.NumMatrices();
        const auto matrixSizeInput = numRows * colDataSizeInput;
        const auto matrixSizeOutput = numRows * colDataSizeOutput;

        T* outputPtr = static_cast<T*>(output.DataPtr);
        const T* inputPtr = static_cast<T*>(input.DataPtr);

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t i = 0; i < numRows; ++i)
                for (std::size_t j = 0; j < numCols; ++j)
                {
                    outputPtr[batchIdx * matrixSizeOutput +
                              i * colDataSizeOutput + j] =
                        inputPtr[batchIdx * matrixSizeInput +
                                 i * colDataSizeInput + j] *
                        toMul;
                }
    }
};
} // namespace CubbyDNN
#endif  // CUBBYDNN_COMPUTETENSOR_HPP
