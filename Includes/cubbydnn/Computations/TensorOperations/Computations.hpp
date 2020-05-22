// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_OPERATIONS_HPP
#define CUBBYDNN_OPERATIONS_HPP

#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Computations/Activations/ActivationFunc.hpp>
#include <memory>


namespace CubbyDNN::Compute
{
void Multiply(const Tensor& inputA, const Tensor& inputB,
              Tensor& output);

void Add(const Tensor& inputA, const Tensor& inputB, Tensor& output);

void Add(Tensor& tensor, const Tensor& toAdd);

//! Adds up all tensors in tensorVector and outputs result to output
void Add(const std::vector<Tensor>& tensorVector, Tensor& output);

//! Returns mean of input tensor from axis
//! This function assumes the highest dimension is for batch
void BatchMean(const Tensor& tensor,  std::size_t idx, Tensor& output);

void Transpose(const Tensor& input, Tensor& output);

void ActivationForward(const Tensor& input, Tensor& output,
                       std::unique_ptr<ActivationFunc>& activation);

void ActivationBackward(const Tensor& input, Tensor& output,
                        std::unique_ptr<ActivationFunc>& activation);

void Dot(const Tensor& inputA, const Tensor& inputB, Tensor& output);

void ScalarMul(const Tensor& input, float toMul, Tensor& output);
} // namespace CubbyDNN
#endif
