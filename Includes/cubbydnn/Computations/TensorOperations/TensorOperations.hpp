// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOROPERATIONS_HPP
#define CUBBYDNN_TENSOROPERATIONS_HPP

#include <cubbydnn/Tensors/Tensor.hpp>

namespace CubbyDNN
{
class TensorOperation
{
public:
    TensorOperation() = default;
    virtual ~TensorOperation() = default;

    TensorOperation(const TensorOperation& other) = default;
    TensorOperation(TensorOperation&& other) noexcept = default;
    TensorOperation& operator=(const TensorOperation& other) = default;
    TensorOperation& operator=(TensorOperation&& other) noexcept = default;
    virtual void Add(const Tensor& inputA, const Tensor& inputB,
                     Tensor& output) = 0;

    virtual void Multiply(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output) = 0;

    virtual void Transpose(const Tensor& input, Tensor& output) = 0;

    virtual void Reshape(const Tensor& input, Tensor& output)
    {
        if (input.Info.GetNumberSystem() != output.Info.GetNumberSystem())
            throw std::runtime_error("Number system mismatch");

        if (input.Info.GetByteSize() != output.Info.GetByteSize())
            throw std::runtime_error("Byte size mismatch between tensors");

        std::memcpy(output.DataPtr, input.DataPtr, input.Info.GetByteSize());
    }

    virtual void Activation(const Tensor& input, Tensor& output) = 0;
};
} // namespace CubbyDNN

#endif
