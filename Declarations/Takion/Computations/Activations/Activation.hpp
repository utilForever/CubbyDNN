// Copyright(c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_ACTIVATION_HPP
#define TAKION_COMPUTE_ACTIVATION_HPP

#include <Takion/Computations/Activations/ActivationDecl.hpp>
#include <cmath>


namespace Takion::Compute
{
template <typename T>
void ReLU<T>::Apply(Tensor<T>& input, Tensor<T>& output) const
{

    const auto inputShape = input.TensorShape;
    const auto outputShape = output.TensorShape;

    const std::size_t colDataSize = input.GetColumnElementSize();
    const std::size_t colDataSizeOutput = output.GetColumnElementSize();

    const auto numRows = inputShape.NumRow();
    const auto numCols = inputShape.NumCol();
    const auto batchSize = inputShape.NumMatrices();
    const auto matrixSize = numRows * colDataSize;
    const auto matrixSizeOutput = numRows * colDataSizeOutput;

    const T* inputPtr = static_cast<T*>(input.DataPtr);
    T* outputPtr = static_cast<T*>(output.DataPtr);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t i = 0; i < numRows; ++i)
            for (std::size_t j = 0; j < numCols; ++j)
            {
                outputPtr[batchIdx * matrixSizeOutput + i * colDataSizeOutput +
                          j] =
                    m_apply(
                        inputPtr[batchIdx * matrixSize + i * colDataSize + j]);
            }
}

template <typename T>
void ReLU<T>::ApplyDerivative(Tensor<T>& input, Tensor<T>& output) const
{
    const auto inputShape = input.TensorShape;
    const auto outputShape = output.TensorShape;

    const std::size_t colDataSize = input.GetColumnElementSize();
    const std::size_t colDataSizeOutput = output.GetColumnElementSize();

    const auto numRows = inputShape.NumRow();
    const auto numCols = inputShape.NumCol();
    const auto batchSize = inputShape.NumMatrices();
    const auto matrixSize = numRows * colDataSize;
    const auto matrixSizeOutput = numRows * colDataSizeOutput;

    const T* inputPtr = static_cast<T*>(input.DataPtr);
    T* outputPtr = static_cast<T*>(output.DataPtr);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (std::size_t i = 0; i < numRows; ++i)
            for (std::size_t j = 0; j < numCols; ++j)
            {
                outputPtr[batchIdx * matrixSizeOutput + i * colDataSizeOutput +
                          j] =
                    m_applyDerivative(
                        inputPtr[batchIdx * matrixSize + i * colDataSize + j]);
            }
}


template <typename T>
void SoftMax<T>::Apply(Tensor& input, Tensor& output) const
{
    const auto inputShape = input.TensorShape;
    const auto outputShape = output.TensorShape;

    const std::size_t colDataSize = input.GetColumnElementSize();
    const std::size_t colDataSizeOutput = output.GetColumnElementSize();

    const auto numRows = inputShape.NumRows();
    const auto numCols = inputShape.NumCols();
    const auto batchSize = inputShape.NumMatrices();
    const auto matrixSize = numRows * colDataSize;
    const auto matrixSizeOutput = numRows * colDataSizeOutput;

    const T* inputPtr = static_cast<T*>(input.DataPtr);
    T* outputPtr = static_cast<T*>(output.DataPtr);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t i = 0; i < numRows; ++i)
        {
            T sum = static_cast<T>(0);

            for (std::size_t j = 0; j < numCols; ++j)
                sum += static_cast<T>(std::exp(
                    inputPtr[batchIdx * matrixSize + i * colDataSize + j]));

            for (std::size_t j = 0; j < numCols; ++j)
            {
                outputPtr[batchIdx * matrixSizeOutput + i * colDataSizeOutput +
                          j] =
                    static_cast<T>(std::exp(
                        inputPtr[batchIdx * matrixSize + i * colDataSize + j]))
                    / sum;
            }
        }
    }
}

template <typename T>
void SoftMax<T>::ApplyDerivative(Tensor& input, Tensor& output) const
{
    const auto inputShape = input.TensorShape;
    const auto outputShape = output.TensorShape;

    const std::size_t colDataSize = input.GetColumnElementSize();
    const std::size_t colDataSizeOutput = output.GetColumnElementSize();

    const auto numRows = inputShape.NumRows();
    const auto numCols = inputShape.NumCols();
    const auto batchSize = inputShape.NumMatrices();
    const auto matrixSize = numRows * colDataSize;
    const auto matrixSizeOutput = numRows * colDataSizeOutput;

    const T* inputPtr = static_cast<T*>(input.DataPtr);
    T* outputPtr = static_cast<T*>(output.DataPtr);

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t i = 0; i < numRows; ++i)
        {
            T sum = static_cast<T>(0);

            for (std::size_t j = 0; j < numCols; ++j)
                sum += static_cast<T>(std::exp(
                    inputPtr[batchIdx * matrixSize + i * colDataSize + j]));

            for (std::size_t j = 0; j < numCols; ++j)
            {
                const auto temp =
                    static_cast<T>(inputPtr[
                        batchIdx * matrixSize + i * colDataSize + j]) / sum;

                outputPtr[batchIdx * matrixSizeOutput + i * colDataSizeOutput +
                          j] = temp - temp * temp;
            }
        }
    }
}
}
#endif
