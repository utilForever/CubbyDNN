// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_MATHKERNEL_HPP
#define TAKION_COMPUTE_MATHKERNEL_HPP

#include <Takion/Computations/GEMM/FloatGem.hpp>
#include <Takion/Computations/GEMM/IntegerGem.hpp>
#include <Takion/Tensors/Tensor.hpp>

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
            CPU::MultiplyCpu(A.Data, B.Data, out.Data, inputShapeA.NumRow(),
                             A.ColumnElementSize(), inputShapeB.NumRow(),
                             B.ColumnElementSize(), out.NumMatrix());
        else if (A.BatchSize == 1)
        {
            CPU::MultiplyWithBroadcastCpu(
                A.Data, B.Data, out.Data, outputShape.NumRow(),
                A.ColumnElementSize(), inputShapeB.NumRow(),
                B.ColumnElementSize(), out.NumMatrix(),
                true);
        }
        else if (B.BatchSize == 1)
        {
            CPU::MultiplyWithBroadcastCpu(
                A.Data, B.Data, out.Data, outputShape.NumRow(),
                A.ColumnElementSize(), inputShapeB.NumRow(),
                B.ColumnElementSize(), out.NumMatrix(),
                false);
        }
        else
        {
            throw std::invalid_argument(
                "Batch size mismatch between given tensors");
        }

        CPU::AddCpu(out.Data, C.Data, out.Data, out.ElementSize(),
                    out.BatchSize);
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
            CPU::MultiplyCpu(A.Data, B.Data, out.Data, outputShape.NumRow(),
                             A.ColumnElementSize(), inputShapeB.NumRow(),
                             B.ColumnElementSize(),
                             out.NumMatrix());
        else if (A.BatchSize == 1)
        {
            CPU::MultiplyWithBroadcastCpu(
                A.Data, B.Data, out.Data, outputShape.NumRow(),
                A.ColumnElementSize(), inputShapeB.NumRow(),
                B.ColumnElementSize(), out.NumMatrix(),
                true);
        }
        else if (B.BatchSize == 1)
        {
            CPU::MultiplyWithBroadcastCpu(
                A.Data, B.Data, out.Data, outputShape.NumRow(),
                A.ColumnElementSize(), inputShapeB.NumRow(),
                B.ColumnElementSize(), out.NumMatrix(),
                false);
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
        CPU::ShrinkCpu(input.Data, output.Data, size, input.BatchSize);
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
        if (A.BatchSize == B.BatchSize)
            CPU::AddCpu(A.Data, B.Data, out.Data, out.ElementSize(),
                        out.BatchSize);
        else if (A.BatchSize == 1)
            CPU::AddWithBroadcastCpu(A.Data, B.Data, out.Data,
                                     out.ElementSize(), out.BatchSize, true);
        else if (B.BatchSize == 1)
            CPU::AddWithBroadcastCpu(A.Data, B.Data, out.Data,
                                     out.ElementSize(), out.BatchSize, false);
        else
            throw std::invalid_argument(
                "Batch size mismatch between given tensors");
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
        if (A.BatchSize == out.BatchSize)
            CPU::AddCpu(out.Data, A.Data, out.Data, out.ElementSize(),
                        out.BatchSize);
        else
            throw std::invalid_argument(
                "Batch size mismatch between given tensors");
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
        if (A.BatchSize == B.BatchSize)
            CPU::SubCpu(A.Data, B.Data, out.Data, out.ElementSize(),
                        out.BatchSize);
        else if (A.BatchSize == 1)
            CPU::SubWithBroadcastCpu(A.Data, B.Data, out.Data,
                                     out.ElementSize(), out.BatchSize, true);
        else if (B.BatchSize == 1)
            CPU::SubWithBroadcastCpu(A.Data, B.Data, out.Data,
                                     out.ElementSize(), out.BatchSize, false);
        else
            throw std::invalid_argument(
                "Batch size mismatch between given tensors");
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
        if (A.BatchSize == out.BatchSize)
            CPU::AddCpu(out.Data, A.Data, out.Data, out.ElementSize(),
                        out.BatchSize);
        else
            throw std::invalid_argument(
                "Batch size mismatch between given tensors");
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
        if (A.BatchSize == B.BatchSize)
            CPU::DotCpu(A.Data, B.Data, out.Data, out.ElementSize(),
                        out.BatchSize);
        else if (A.BatchSize == 1)
            CPU::DotWithBroadcastCpu(A.Data, B.Data, out.Data,
                                     out.ElementSize(),
                                     out.BatchSize, true);
        else if (B.BatchSize == 1)
            CPU::DotWithBroadcastCpu(A.Data, B.Data, out.Data,
                                     out.ElementSize(),
                                     out.BatchSize, false);
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
        CPU::DotCpu(out.Data, in.Data, out.Data, out.ElementSize(),
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
        CPU::ScalarMulCpu(in.Data, toMul, out.Data, out.ElementSize(),
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
        CPU::ScalarMulCpu(tensor.Data, toMul, tensor.Data, tensor.ElementSize(),
                          tensor.BatchSize);
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
        CPU::ScalarDivCpu(in.Data, toDiv, out.Data, out.ElementSize(),
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
        CPU::ScalarDivCpu(tensor.Data, toDiv, tensor.Data, tensor.ElementSize(),
                          tensor.BatchSize);
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
        CPU::SetCpu(tensor.Data, toSet, tensor.ElementSize(), tensor.BatchSize);
    }
    else
        throw std::runtime_error("Not implemented");
}

template <typename T, typename Function>
void Apply(const Tensor<T>& input, Tensor<T>& output, Function lambda)
{
    const auto device = input.Device;
    if (device.Type() == DeviceType::CPU)
    {
        CPU::ApplyCpu(input.Data, output.Data, lambda, output.ElementSize(),
                      output.BatchSize);
    }
    else
        throw std::runtime_error("Not implemented");
}
}

#endif
