// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTETENSOR_HPP
#define CUBBYDNN_COMPUTETENSOR_HPP

#include <algorithm>
#include <cubbydnn/Tensors/Tensor.hpp>
#include <functional>

namespace CubbyDNN
{
class Naive
{
public:
    template <typename T>
    static void BasicLoop(const Tensor& input, Tensor& output,
                          const std::function<T(T&)>& function)
    {
        const auto inputShape = input.TensorShape;
        const auto outputShape = output.TensorShape;

        const auto batchSize = inputShape.BatchSize();
        const auto batchOutputSize = outputShape.BatchSize();

        const auto numRows = inputShape.NumRows();
        const auto numCols = inputShape.NumCols();

        if (batchSize != batchOutputSize)
            throw std::runtime_error("TensorMul - batch size mismatch");

        for (std::size_t batch = 0; batch < batchSize; ++batch)
            for (std::size_t i = 0; i < numRows; ++i)
                for (std::size_t j = 0; j < numCols; ++j)
                    static_cast<T*>(output.DataPtr)[i * numCols + j] = function(
                        static_cast<T*>(input.DataPtr)[i * numCols + j]);
    }

    template <typename T>
    static void TensorAdd(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output)
    {
        const auto inputShapeA = inputA.TensorShape;
        const auto inputShapeB = inputB.TensorShape;
        const auto outputShape = output.TensorShape;

        const auto batchSizeA = inputShapeA.BatchSize();
        const auto batchSizeB = inputShapeB.BatchSize();
        const auto batchOutputSize = outputShape.BatchSize();

        const auto numRowsA = inputShapeA.NumRows();
        const auto numColsA = inputShapeA.NumCols();

        const auto numRowsB = inputShapeB.NumRows();
        const auto numColsB = inputShapeB.NumCols();

        if (batchSizeA != batchSizeB || batchSizeA != batchOutputSize)
            throw std::runtime_error("TensorMul - batch size mismatch");

        if (numRowsB != numRowsA || numColsB != numColsA ||
            numRowsB != outputShape.NumRows() || numColsB != outputShape.
            NumCols())
            throw std::runtime_error("TensorMul - matrix shape mismatch");

        for (std::size_t batch = 0; batch < batchSizeA; ++batch)
            for (std::size_t i = 0; i < numRowsA; ++i)
                for (std::size_t j = 0; j < numColsA; ++j)
                    static_cast<T*>(output.DataPtr)[i * numColsA + j] =
                        static_cast<T*>(inputA.DataPtr)[i * numColsA + j] +
                        static_cast<T*>(inputB.DataPtr)[i * numColsA + j];
    }

    template <typename T>
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

        const auto numRowsA = inputShapeA.NumRows();
        const auto numColsA = inputShapeA.NumCols();

        const auto numRowsB = inputShapeB.NumRows();
        const auto numColsB = inputShapeB.NumCols();

        const auto blockSize = 16;

        //! Optimized matrix multiplication minimizing cache misses
        for (std::size_t batch = 0; batch < batchSizeA; ++batch)
            for (std::size_t jj = 0; jj < numColsB; jj = jj + blockSize)
                for (std::size_t kk = 0; kk < numColsA; kk = kk + blockSize)
                    for (std::size_t i = 0; i < numRowsA; ++i)
                        for (std::size_t j = jj;
                             j < std::min(jj + blockSize, numColsB); ++j)
                        {
                            T r = static_cast<T>(0);
                            for (std::size_t k = kk;
                                 k < std::min(kk + blockSize, numColsA); ++k)
                            {
                                r += static_cast<T*>(
                                        inputA.DataPtr)[i * numRowsA + k] *
                                    static_cast<T*>(
                                        inputB.DataPtr)[k * numRowsB + j];
                            }
                            static_cast<T*>(output.DataPtr)[i * numRowsA + j] +=
                                r;
                        }
    }

    template <typename T>
    static void TensorTranspose(const Tensor& input, Tensor& output)
    {
        const auto inputShape = input.TensorShape;
        const auto outputShape = output.TensorShape;

        const auto batchSize = inputShape.BatchSize();
        const auto batchOutputSize = outputShape.BatchSize();

        const auto numRows = inputShape.NumRows();
        const auto numCols = inputShape.NumCols();

        if (batchSize != batchOutputSize)
            throw std::runtime_error("TensorTranspose - batch size mismatch");;

        if (numRows != outputShape.NumCols() || numCols != outputShape.NumRows()
        )
            throw std::runtime_error("TensorTranspose - matrix shape mismatch");

        const auto blockSize = 16;
        //! Optimized matrix transpose minimizing cache misses
        for (std::size_t batch = 0; batch < batchSize; ++batch)
            for (std::size_t ii = 0; ii < numCols; ii = ii + blockSize)
                for (std::size_t jj = 0; jj < numRows; jj = jj + blockSize)
                    for (std::size_t i = 0;
                         i < std::min(numCols, ii + blockSize); ++i)
                        for (std::size_t j = jj;
                             j < std::min(numRows, jj + blockSize); ++j)
                        {
                            static_cast<T*>(output.DataPtr)[numRows * i + j] =
                                static_cast<T*>(input.DataPtr)[numCols * j + i];
                        }
    }
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTETENSOR_HPP
