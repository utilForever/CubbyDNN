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
void Multiply(const Tensor<T>& A, const Tensor<T>& B,
              Tensor<T>& out)
{
    const auto numMatrices = out.NumMatrix();
    const auto numRow = out.TensorShape.NumRow();
    const auto numCol = out.TensorShape.NumCol();
    const auto numMiddle = A.TensorShape.NumCol();
    const auto device = out.Device;

    if (device.Type() == Compute::DeviceType::CPU)
    {
        if (A.BatchSize == B.BatchSize)
        {
            for (std::size_t batchIdx = 0; batchIdx < numMatrices; ++batchIdx)
            {
                for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
                    {
                        T sum = static_cast<T>(0);
                        for (std::size_t midIdx = 0; midIdx < numMiddle;
                             ++midIdx)
                            sum += A.At(batchIdx, { rowIdx, midIdx }) *
                                B.At(batchIdx, { midIdx, colIdx });

                        out.At(batchIdx, { rowIdx, colIdx }) = sum;
                    }
            }
        }
        else if (A.BatchSize == 1)
        {
            for (std::size_t batchIdx = 0; batchIdx < numMatrices; ++batchIdx)
            {
                for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
                    {
                        T sum = static_cast<T>(0);
                        for (std::size_t midIdx = 0; midIdx < numMiddle;
                             ++midIdx)
                            sum += A.At(0, { rowIdx, midIdx }) *
                                B.At(batchIdx, { midIdx, colIdx });

                        out.At(batchIdx, { rowIdx, colIdx }) = sum;
                    }
            }
        }
        else if (B.BatchSize == 1)
        {
            for (std::size_t batchIdx = 0; batchIdx < numMatrices; ++batchIdx)
            {
                for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
                    for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
                    {
                        T sum = static_cast<T>(0);
                        for (std::size_t midIdx = 0; midIdx < numMiddle;
                             ++midIdx)
                            sum += A.At(batchIdx, { rowIdx, midIdx }) *
                                B.At(0, { midIdx, colIdx });

                        out.At(batchIdx, { rowIdx, colIdx }) = sum;
                    }
            }
        }
        else
        {
            throw std::invalid_argument(
                "Batch size mismatch between given tensors");
        }
    }
}

template <typename T>
void Transpose(const Tensor<T>& in, Tensor<T>& out)
{
    const auto matSize = out.NumMatrix();
    const auto inputShape = in.TensorShape;
    const auto numRow = inputShape.NumRow();
    const auto numCol = inputShape.NumCol();

    for (std::size_t matIdx = 0; matIdx < matSize; ++matIdx)
    {
        const auto matOffset = numRow * numCol * matIdx;
        for (std::size_t rowIdx = 0; rowIdx < numRow; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < numCol; ++colIdx)
            {
                out.At(matOffset + numCol * rowIdx + colIdx) =
                    in.At(matOffset + numRow * colIdx + rowIdx);
            }
    }
}

template <typename T>
void Shrink(const Tensor<T>& in, Tensor<T>& out)
{
    const auto batchSize = in.BatchSize;
    const auto elementSize = out.TensorShape.Size();

    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        const auto batchOffset = elementSize * batchIdx;
        for (std::size_t idx = 0; idx < elementSize; ++idx)
            out.At(idx) += in.At(batchOffset + idx);
    }

    for (std::size_t idx = 0; idx < elementSize; ++idx)
        out.At(idx) /= static_cast<T>(batchSize);
}

template <typename T>
void Add(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& out)
{
    const auto batchSize = out.BatchSize;
    const auto elementSize = out.TensorShape.Size();
    const auto device = out.Device;

    if (device.Type() == Compute::DeviceType::CPU)
    {
        if (A.BatchSize == B.BatchSize)
        {
            for (std::size_t idx = 0; idx < batchSize * elementSize; ++idx)
            {
                const auto sum = A.At(idx) + B.At(idx);
                out.At(idx) = sum;
            }
        }
        else if (A.BatchSize == 1)
        {
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const auto batchOffset = batchIdx * elementSize;
                for (std::size_t idx = 0; idx < elementSize; ++idx)
                {
                    const auto sum = A.At(idx) + B.At(batchOffset + idx);
                    out.At(batchOffset + idx) = sum;
                }
            }
        }
        else if (B.BatchSize == 1)
        {
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const auto batchOffset = batchIdx * elementSize;
                for (std::size_t idx = 0; idx < elementSize; ++idx)
                {
                    const auto sum = A.At(batchOffset + idx) + B.At(idx);
                    out.At(batchOffset + idx) = sum;
                }
            }
        }
    }
    else
    {
        throw std::invalid_argument("Not implemented");
    }
}

template <typename T>
void Sub(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& out)
{
    const auto batchSize = out.BatchSize;
    const auto elementSize = out.TensorShape.Size();
    const auto device = out.Device;

    if (device.Type() == Compute::DeviceType::CPU)
    {
        if (A.BatchSize == B.BatchSize)
        {
            for (std::size_t idx = 0; idx < batchSize * elementSize; ++idx)
            {
                const auto sum = A.At(idx) - B.At(idx);
                out.At(idx) = sum;
            }
        }
        else if (A.BatchSize == 1)
        {
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const auto batchOffset = batchSize * elementSize;
                for (std::size_t idx = 0; idx < elementSize; ++idx)
                {
                    const auto sum = A.At(idx) - B.At(batchOffset + idx);
                    out.At(batchOffset + idx) = sum;
                }
            }
        }
        else if (B.BatchSize == 1)
        {
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const auto batchOffset = batchSize * elementSize;
                for (std::size_t idx = 0; idx < elementSize; ++idx)
                {
                    const auto sum = A.At(batchOffset + idx) - B.At(idx);
                    out.At(batchOffset + idx) = sum;
                }
            }
        }
    }
    else
    {
        throw std::invalid_argument("Not implemented");
    }
}

template <typename T>
void Dot(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& out)
{
    const auto batchSize = out.BatchSize;
    const auto elementSize = out.TensorShape.Size();
    const auto device = out.Device;

    if (device.Type() == Compute::DeviceType::CPU)
    {
        if (A.BatchSize == B.BatchSize)
        {
            for (std::size_t idx = 0; idx < batchSize * elementSize; ++idx)
            {
                const auto sum = A.At(idx) * B.At(idx);
                out.At(idx) = sum;
            }
        }
        else if (A.BatchSize == 1)
        {
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const auto batchOffset = batchIdx * elementSize;
                for (std::size_t idx = 0; idx < elementSize; ++idx)
                {
                    const auto sum = A.At(idx) * B.At(batchOffset + idx);
                    out.At(batchOffset + idx) = sum;
                }
            }
        }
        else if (B.BatchSize == 1)
        {
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const auto batchOffset = batchIdx * elementSize;
                for (std::size_t idx = 0; idx < elementSize; ++idx)
                {
                    const auto sum = A.At(batchOffset + idx) * B.At(idx);
                    out.At(batchOffset + idx) = sum;
                }
            }
        }
    }
    else
    {
        throw std::invalid_argument("Not implemented");
    }
}

template <typename T>
void ScalarMul(const Tensor<T>& in, T toMul, Tensor<T>& out)
{
    const auto batchSize = out.BatchSize;
    const auto elementSize = out.TensorShape.Size();
    const auto device = out.Device;

    for (std::size_t idx = 0; idx < batchSize * elementSize; ++idx)
    {
        const auto mul = in.At(idx) * toMul;
        out.At(idx) = mul;
    }
}

template <typename T>
void ScalarDiv(const Tensor<T>& in, T toDiv, Tensor<T>& out)
{
    const auto batchSize = out.BatchSize;
    const auto elementSize = out.TensorShape.Size();
    const auto device = out.Device;

    for (std::size_t idx = 0; idx < batchSize * elementSize; ++idx)
    {
        const auto div = in.At(idx) * toDiv;
        out.At(idx) = div;
    }
}

template <typename T>
void Set(Tensor<T>& tensor, T toSet)
{
    const auto batchSize = tensor.BatchSize;
    const auto elementSize = tensor.TensorShape.Size();
    const auto device = tensor.Device;

    for (std::size_t idx = 0; idx < batchSize * elementSize; ++idx)
    {
        tensor.At(idx) = toSet;
    }
}

template <typename T, typename Function>
void Apply(const Tensor<T>& in, Tensor<T>& out, Function lambda)
{
    const auto batchSize = out.BatchSize;
    const auto elementSize = out.TensorShape.Size();
    const auto device = out.Device;

    for (std::size_t idx = 0; idx < batchSize * elementSize; ++idx)
    {
        const auto mul = lambda(in.At(idx));
        out.At(idx) = mul;
    }
}
}

#endif
