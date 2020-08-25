// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_MATHKERNEL_HPP
#define TAKION_COMPUTE_MATHKERNEL_HPP

#include <Takion/Computations/GEMM/FloatGemm.hpp>
#include <Takion/Computations/GEMM/IntegerGemm.hpp>
#include <Takion/Tensors/Tensor.hpp>
#include <type_traits>

namespace Takion::Compute
{
template <typename T>
void MultiplyAdd(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C,
                 Tensor<T>& out)
{
    const auto device = out.Device;
    const auto outputShape = out.TensorShape;
    const auto inputShapeA = A.TensorShape;
    const auto inputShapeB = B.TensorShape;

    if (device.Type() == DeviceType::CPU)
    {
        if (A.BatchSize == B.BatchSize)
        {
            if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
            {
                CPU::Float::MultiplyCpu(
                    A.Data, B.Data, out.Data, inputShapeA.NumRow(),
                    A.ColumnElementSize(), inputShapeB.NumRow(),
                    B.ColumnElementSize(), out.NumMatrix());

                CPU::Float::AddCpu(out.Data, C.Data, out.Data,
                                   out.ElementSize(), out.BatchSize);
            }
            else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
            {
                CPU::Int::MultiplyCpu(
                    A.Data, B.Data, out.Data, inputShapeA.NumRow(),
                    A.ColumnElementSize(), inputShapeB.NumRow(),
                    B.ColumnElementSize(), out.NumMatrix());

                CPU::Int::AddCpu(out.Data, C.Data, out.Data,
                                 out.ElementSize(), out.BatchSize);
            }
        }
        else if (A.BatchSize == 1)
        {
            if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
            {
                CPU::Float::MultiplyWithBroadcastCpu(
                    A.Data, B.Data, out.Data, outputShape.NumRow(),
                    A.ColumnElementSize(), inputShapeB.NumRow(),
                    B.ColumnElementSize(), out.NumMatrix(), true);

                CPU::Int::MultiplyWithBroadcastCpu(
                    A.Data, B.Data, out.Data, outputShape.NumRow(),
                    A.ColumnElementSize(), inputShapeB.NumRow(),
                    B.ColumnElementSize(), out.NumMatrix(), true);
            }
            else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
            {
                CPU::Int::MultiplyWithBroadcastCpu(
                    A.Data, B.Data, out.Data, outputShape.NumRow(),
                    A.ColumnElementSize(), inputShapeB.NumRow(),
                    B.ColumnElementSize(), out.NumMatrix(), true);

                CPU::Int::AddWithBroadcastCpu(out.Data, C.Data, out.Data,
                                              out.ElementSize(), out.BatchSize,
                                              true);
            }
        }
        else if (B.BatchSize == 1)
        {
            if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
            {
                CPU::Float::MultiplyWithBroadcastCpu(
                    A.Data, B.Data, out.Data, outputShape.NumRow(),
                    A.ColumnElementSize(), inputShapeB.NumRow(),
                    B.ColumnElementSize(), out.NumMatrix(), false);

                CPU::Float::AddWithBroadcastCpu(out.Data, C.Data, out.Data,
                                                out.ElementSize(),
                                                out.BatchSize, false);
            }
            else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
            {
                CPU::Int::MultiplyWithBroadcastCpu(
                    A.Data, B.Data, out.Data, outputShape.NumRow(),
                    A.ColumnElementSize(), inputShapeB.NumRow(),
                    B.ColumnElementSize(), out.NumMatrix(), false);

                CPU::Int::AddWithBroadcastCpu(out.Data, C.Data, out.Data,
                                              out.ElementSize(), out.BatchSize,
                                              false);
            }
        }
        else
        {
            throw std::invalid_argument(
                "Batch size mismatch between given tensors");
        }
    }
    else
    {
        std::runtime_error("Not implemented");
    }
}

template <typename T>
void Multiply(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& out)
{
    const auto device = out.Device;
    const auto outputShape = out.TensorShape;
    const auto inputShapeA = A.TensorShape;
    const auto inputShapeB = B.TensorShape;;

    if (device.Type() == DeviceType::CPU)
    {
        if (A.BatchSize == B.BatchSize)
        {
            if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
                CPU::Float::MultiplyCpu(
                    A.Data, B.Data, out.Data, inputShapeA.NumRow(),
                    A.ColumnElementSize(), inputShapeB.NumRow(),
                    B.ColumnElementSize(), out.NumMatrix());
            else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
                CPU::Int::MultiplyCpu(
                    A.Data, B.Data, out.Data, inputShapeA.NumRow(),
                    A.ColumnElementSize(), inputShapeB.NumRow(),
                    B.ColumnElementSize(), out.NumMatrix());
        }
        else if (A.BatchSize == 1)
        {
            if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
                CPU::Float::MultiplyWithBroadcastCpu(
                    A.Data, B.Data, out.Data, outputShape.NumRow(),
                    A.ColumnElementSize(), inputShapeB.NumRow(),
                    B.ColumnElementSize(), out.NumMatrix(), true);
            else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
                CPU::Int::MultiplyWithBroadcastCpu(
                    A.Data, B.Data, out.Data, outputShape.NumRow(),
                    A.ColumnElementSize(), inputShapeB.NumRow(),
                    B.ColumnElementSize(), out.NumMatrix(), true);
        }
        else if (B.BatchSize == 1)
        {
            if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
                CPU::Float::MultiplyWithBroadcastCpu(
                    A.Data, B.Data, out.Data, outputShape.NumRow(),
                    A.ColumnElementSize(), inputShapeB.NumRow(),
                    B.ColumnElementSize(), out.NumMatrix(), false);
            else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
                CPU::Int::MultiplyWithBroadcastCpu(
                    A.Data, B.Data, out.Data, outputShape.NumRow(),
                    A.ColumnElementSize(), inputShapeB.NumRow(),
                    B.ColumnElementSize(), out.NumMatrix(), false);
        }
        else
        {
            throw std::invalid_argument(
                "Batch size mismatch between given tensors");
        }
    }
    else
    {
        throw std::runtime_error("Not implemented");
    }
}

template <typename T>
void Transpose(const Tensor<T>& in, Tensor<T>& out)
{
    const auto matSize = out.NumMatrix();
    const auto inputShape = in.TensorShape;
    const auto numRow = inputShape.NumRow();
    const auto numCol = inputShape.NumCol();

#pragma omp parallel for schedule(static) default(shared)
    for (long matIdx = 0; static_cast<std::size_t>(matIdx) < matSize; ++matIdx)
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
void Shrink(const Tensor<T>& input, Tensor<T>& output)
{
    const auto device = output.Device;
    const auto size = output.ElementSize();
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
            CPU::Float::ShrinkCpu(input.Data, output.Data, size,
                                  input.BatchSize);
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
            CPU::Int::ShrinkCpu(input.Data, output.Data, size,
                                input.BatchSize);
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T>
void Add(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& out)
{
    const auto device = out.Device;
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
        {
            if (A.BatchSize == B.BatchSize)
                CPU::Float::AddCpu(A.Data, B.Data, out.Data, out.ElementSize(),
                                   out.BatchSize);
            else if (A.BatchSize == 1)
                CPU::Float::AddWithBroadcastCpu(A.Data, B.Data, out.Data,
                                                out.ElementSize(),
                                                out.BatchSize,
                                                true);
            else if (B.BatchSize == 1)
                CPU::Float::AddWithBroadcastCpu(A.Data, B.Data, out.Data,
                                                out.ElementSize(),
                                                out.BatchSize,
                                                false);
            else
                throw std::invalid_argument(
                    "Batch size mismatch between given tensors");
        }
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
        {
            if (A.BatchSize == B.BatchSize)
                CPU::Int::AddCpu(A.Data, B.Data, out.Data, out.ElementSize(),
                                 out.BatchSize);
            else if (A.BatchSize == 1)
                CPU::Int::AddWithBroadcastCpu(A.Data, B.Data, out.Data,
                                              out.ElementSize(),
                                              out.BatchSize, true);
            else if (B.BatchSize == 1)
                CPU::Int::AddWithBroadcastCpu(A.Data, B.Data, out.Data,
                                              out.ElementSize(),
                                              out.BatchSize, false);
            else
                throw std::invalid_argument(
                    "Batch size mismatch between given tensors");
        }
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T>
void Add(const Tensor<T>& A, Tensor<T>& out)
{
    const auto device = out.Device;
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
        {
            if (A.BatchSize == out.BatchSize)
                CPU::Float::AddCpu(out.Data, A.Data, out.Data,
                                   out.ElementSize(),
                                   out.BatchSize);
            else
                throw std::invalid_argument(
                    "Batch size mismatch between given tensors");
        }
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
        {
            if (A.BatchSize == out.BatchSize)
                CPU::Int::AddCpu(out.Data, A.Data, out.Data, out.ElementSize(),
                                 out.BatchSize);
            else
                throw std::invalid_argument(
                    "Batch size mismatch between given tensors");
        }
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T>
void Sub(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& out)
{
    const auto device = out.Device;
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
        {
            if (A.BatchSize == B.BatchSize)
                CPU::Float::SubCpu(A.Data, B.Data, out.Data, out.ElementSize(),
                                   out.BatchSize);
            else if (A.BatchSize == 1)
                CPU::Float::SubWithBroadcastCpu(A.Data, B.Data, out.Data,
                                                out.ElementSize(),
                                                out.BatchSize,
                                                true);
            else if (B.BatchSize == 1)
                CPU::Float::SubWithBroadcastCpu(A.Data, B.Data, out.Data,
                                                out.ElementSize(),
                                                out.BatchSize,
                                                false);
            else
                throw std::invalid_argument(
                    "Batch size mismatch between given tensors");
        }
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
        {
            if (A.BatchSize == B.BatchSize)
                CPU::Int::SubCpu(A.Data, B.Data, out.Data, out.ElementSize(),
                                 out.BatchSize);
            else if (A.BatchSize == 1)
                CPU::Int::SubWithBroadcastCpu(A.Data, B.Data, out.Data,
                                              out.ElementSize(), out.BatchSize,
                                              true);
            else if (B.BatchSize == 1)
                CPU::Int::SubWithBroadcastCpu(A.Data, B.Data, out.Data,
                                              out.ElementSize(), out.BatchSize,
                                              false);
            else
                throw std::invalid_argument(
                    "Batch size mismatch between given tensors");
        }
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T>
void Sub(const Tensor<T>& A, Tensor<T>& out)
{
    const auto device = out.Device;
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
        {
            if (A.BatchSize == out.BatchSize)
                CPU::Float::SubCpu(out.Data, A.Data, out.Data,
                                   out.ElementSize(),
                                   out.BatchSize);
            else
                throw std::invalid_argument(
                    "Batch size mismatch between given tensors");
        }
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
        {
            if (A.BatchSize == out.BatchSize)
                CPU::Int::SubCpu(out.Data, A.Data, out.Data,
                                 out.ElementSize(), out.BatchSize);
            else
                throw std::invalid_argument(
                    "Batch size mismatch between given tensors");
        }
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T>
void Dot(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& out)
{
    const auto device = out.Device;
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
        {
            if (A.BatchSize == B.BatchSize)
                CPU::Float::DotCpu(A.Data, B.Data, out.Data, out.ElementSize(),
                                   out.BatchSize);
            else if (A.BatchSize == 1)
                CPU::Float::DotWithBroadcastCpu(A.Data, B.Data, out.Data,
                                                out.ElementSize(),
                                                out.BatchSize,
                                                true);
            else if (B.BatchSize == 1)
                CPU::Float::DotWithBroadcastCpu(A.Data, B.Data, out.Data,
                                                out.ElementSize(),
                                                out.BatchSize,
                                                false);
        }
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
        {
            if (A.BatchSize == B.BatchSize)
                CPU::Int::DotCpu(A.Data, B.Data, out.Data, out.ElementSize(),
                                 out.BatchSize);
            else if (A.BatchSize == 1)
                CPU::Int::DotWithBroadcastCpu(A.Data, B.Data, out.Data,
                                              out.ElementSize(),
                                              out.BatchSize, true);
            else if (B.BatchSize == 1)
                CPU::Int::DotWithBroadcastCpu(A.Data, B.Data, out.Data,
                                              out.ElementSize(),
                                              out.BatchSize, false);
        }
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T>
void Dot(const Tensor<T>& in, Tensor<T>& out)
{
    const auto device = out.Device;
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
        {
            CPU::Float::DotCpu(out.Data, in.Data, out.Data, out.ElementSize(),
                               out.BatchSize);
        }
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
        {
            CPU::Int::DotCpu(out.Data, in.Data, out.Data, out.ElementSize(),
                             out.BatchSize);
        }
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T>
void Div(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& out)
{
    const auto device = out.Device;
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
        {
            if (A.BatchSize == B.BatchSize)
                CPU::Float::DivCpu(A.Data, B.Data, out.Data, out.ElementSize(),
                                   out.BatchSize);
            else if (A.BatchSize == 1)
                CPU::Float::DivWithBroadcastCpu(A.Data, B.Data, out.Data,
                                                out.ElementSize(),
                                                out.BatchSize,
                                                true);
            else if (B.BatchSize == 1)
                CPU::Float::DivWithBroadcastCpu(A.Data, B.Data, out.Data,
                                                out.ElementSize(),
                                                out.BatchSize,
                                                false);
        }
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
        {
            if (A.BatchSize == B.BatchSize)
                CPU::Int::DivCpu(A.Data, B.Data, out.Data, out.ElementSize(),
                                 out.BatchSize);
            else if (A.BatchSize == 1)
                CPU::Int::DivWithBroadcastCpu(A.Data, B.Data, out.Data,
                                              out.ElementSize(),
                                              out.BatchSize, true);
            else if (B.BatchSize == 1)
                CPU::Int::DivWithBroadcastCpu(A.Data, B.Data, out.Data,
                                              out.ElementSize(),
                                              out.BatchSize, false);
        }
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T>
void Div(const Tensor<T>& in, Tensor<T>& out)
{
    const auto device = out.Device;
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
            CPU::Float::DivCpu(out.Data, in.Data, out.Data, out.ElementSize(),
                               out.BatchSize);
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
            CPU::Int::DivCpu(out.Data, in.Data, out.Data, out.ElementSize(),
                             out.BatchSize);
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T>
void ScalarMul(const Tensor<T>& in, T toMul, Tensor<T>& out)
{
    const auto device = out.Device;
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
            CPU::Float::ScalarMulCpu(in.Data, toMul, out.Data,
                                     out.ElementSize(),
                                     out.BatchSize);
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
            CPU::Int::ScalarMulCpu(in.Data, toMul, out.Data, out.ElementSize(),
                                   out.BatchSize);
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T>
void ScalarMul(const Tensor<T>& tensor, T toMul)
{
    const auto device = tensor.Device;
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
            CPU::Float::ScalarMulCpu(tensor.Data, toMul, tensor.Data,
                                     tensor.ElementSize(),
                                     tensor.BatchSize);
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
            CPU::Int::ScalarMulCpu(tensor.Data, toMul, tensor.Data,
                                   tensor.ElementSize(), tensor.BatchSize);
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T>
void ScalarDiv(const Tensor<T>& in, T toDiv, Tensor<T>& out)
{
    const auto device = out.Device;
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
            CPU::Float::ScalarDivCpu(in.Data, toDiv, out.Data,
                                     out.ElementSize(),
                                     out.BatchSize);
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
            CPU::Int::ScalarDivCpu(in.Data, toDiv, out.Data, out.ElementSize(),
                                   out.BatchSize);
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T>
void ScalarDiv(Tensor<T>& tensor, T toDiv)
{
    const auto device = tensor.Device;
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
            CPU::Float::ScalarDivCpu(tensor.Data, toDiv, tensor.Data,
                                     tensor.ElementSize(),
                                     tensor.BatchSize);
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
            CPU::Int::ScalarDivCpu(tensor.Data, toDiv, tensor.Data,
                                   tensor.ElementSize(), tensor.BatchSize);
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T>
void Set(Tensor<T>& tensor, T toSet)
{
    const auto device = tensor.Device;
    if (device.Type() == DeviceType::CPU)
    {
        if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4)
            CPU::Float::SetCpu(tensor.Data, toSet, tensor.ElementSize(),
                               tensor.BatchSize);
        else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
            CPU::Int::SetCpu(tensor.Data, toSet, tensor.ElementSize(),
                             tensor.BatchSize);
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T, typename Function>
void Apply(const Tensor<T>& input, Tensor<T>& output, Function lambda)
{
    const auto device = input.Device;
    const auto size = output.ElementSize();
    const auto batchSize = output.BatchSize;
#pragma omp parallel for schedule(static) default(shared)
    for (long batchIdx = 0; batchIdx < static_cast<long>(batchSize); batchIdx++)
    {
        const auto batchOffset = size * batchIdx;
        for (unsigned i = 0; i < size; i += 1)
        {
            output.Data[batchOffset + i] =
                static_cast<T>(lambda(input.Data[batchOffset + i]));
        }
    }
}
}

#endif
