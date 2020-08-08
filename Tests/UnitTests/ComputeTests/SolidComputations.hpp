// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_TEST_SOLIDCOMPUTATIONS_HPP
#define TAKION_TEST_SOLIDCOMPUTATIONS_HPP
#include <Takion/Tensors/Tensor.hpp>

namespace Takion::Test
{
template <typename T>
void Multiply(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& out)
{
    const auto batchSize = out.BatchSize;
    const auto numRow = out.TensorShape.NumRow();
    const auto numCol = out.TensorShape.NumCol();
    const auto numMiddle = A.TensorShpae.NumCol();
    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++ batchIdx)
    {
        for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
            {
                T sum = static_cast<T>(0);
                for (std::size_t midIdx = 0; midIdx < numMiddle; ++midIdx)
                    sum += A.At(batchIdx, { rowIdx, midIdx }) *
                        B.At(batchIdx, { midIdx, colIdx });

                out.At(batchIdx, { rowIdx, colIdx }) = sum;
            }
    }
}


template <typename T>
void Transpose(const Tensor<T>& input, Tensor<T>& output)
{
    const auto batchSize = output.BatchSize;
    const auto inputShape = input.TensorShape;
    const auto numRow = inputShape.NumRow();
    const auto numCol = inputShape.NumCol();

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
            {
                output.At(batchIdx, { colIdx, rowIdx }) =
                    input.At(batchIdx, { rowIdx, colIdx });
            }
    }
}

template <typename T>
void Shrink(const Tensor<T>& input, Tensor<T>& output)
{
    const auto batchSize = output.BatchSize();
    const auto elementSize = output.ElementSize;

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t idx = 0; idx < elementSize; ++idx)
            output.Data[idx] += input.Data[idx];
    }

    for (std::size_t idx = 0; idx < elementSize; ++idx)
        output.Data[idx] /= static_cast<T>(batchSize);
}

template <typename T>
void Add(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& out)
{
    const auto batchSize = out.BatchSize;
    const auto numRow = out.TensorShape.NumRow();
    const auto numCol = out.TensorShape.NumCol();

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
            {
                const auto sum = A.At(batchIdx, { rowIdx, colIdx }) +
                                 B.At(batchIdx, { rowIdx, colIdx });

                out.At(batchIdx, { rowIdx, colIdx }) = sum;
            }
    }
}

template <typename T>
void Sub(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& out)
{
    const auto batchSize = out.BatchSize;
    const auto numRow = out.TensorShape.NumRow();
    const auto numCol = out.TensorShape.NumCol();

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
            {
                const auto sub = A.At(batchIdx, { rowIdx, colIdx }) -
                                 B.At(batchIdx, { rowIdx, colIdx });

                out.At(batchIdx, { rowIdx, colIdx }) = sub;
            }
    }
}

template <typename T>
void Dot(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& out)
{
    const auto batchSize = out.BatchSize;
    const auto numRow = out.TensorShape.NumRow();
    const auto numCol = out.TensorShape.NumCol();

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
            {
                const auto dot = A.At(batchIdx, { rowIdx, colIdx }) *
                                 B.At(batchIdx, { rowIdx, colIdx });

                out.At(batchIdx, { rowIdx, colIdx }) = dot;
            }
    }
}

template <typename T>
void ScalarMul(const Tensor<T>& in, T toMul, Tensor<T>& out)
{
    const auto batchSize = out.BatchSize;
    const auto numRow = out.TensorShape.NumRow();
    const auto numCol = out.TensorShape.NumCol();

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
            {
                out.At(batchIdx, { rowIdx, colIdx }) =
                    in.At(batchSize, { rowIdx, colIdx }) * toMul;
            }
    }
}

template <typename T>
void ScalarDiv(const Tensor<T>& in, T toDiv, Tensor<T>& out)
{
    const auto batchSize = out.BatchSize;
    const auto numRow = out.TensorShape.NumRow();
    const auto numCol = out.TensorShape.NumCol();

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
            {
                out.At(batchIdx, { rowIdx, colIdx }) =
                    in.At(batchSize, { rowIdx, colIdx }) * toDiv;
            }
    }
}

template <typename T>
void Set(Tensor<T>& tensor, T toSet)
{
    const auto batchSize = tensor.BatchSize();
    const auto elementSize = tensor.ElementSize;

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t idx = 0; idx < elementSize; ++idx)
            tensor.Data[idx] = toSet;
    }
}

template <typename T, typename Function>
void Apply(const Tensor<T>& input, Tensor<T>& output, Function lambda)
{
    const auto batchSize = input.BatchSize();
    const auto elementSize = input.ElementSize;

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (std::size_t idx = 0; idx < elementSize; ++idx)
            output.Data[idx] = lambda(input);
    }
}
}

#endif
