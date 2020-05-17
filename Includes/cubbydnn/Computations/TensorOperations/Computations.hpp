// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_OPERATIONS_HPP
#define CUBBYDNN_OPERATIONS_HPP

#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Computations/Activations/ActivationFunc.hpp>

namespace CubbyDNN::Compute
{
void Multiply(const Tensor& inputA, const Tensor& inputB,
                     Tensor& output);

void Add(const Tensor& inputA, const Tensor& inputB, Tensor& output);

void Transpose(const Tensor& input, Tensor& output);

void ActivationForward(const Tensor& input, Tensor& output,
                              std::unique_ptr<ActivationFunc>& activation);

void ActivationBackward(const Tensor& input, Tensor& output,
                               std::unique_ptr<ActivationFunc>& activation);

void Dot(const Tensor& inputA, const Tensor& inputB, Tensor& output);

} // namespace CubbyDNN
#endif
