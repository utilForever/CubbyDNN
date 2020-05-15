// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_NAIVEOPERATIONS_HPP
#define CUBBYDNN_NAIVEOPERATIONS_HPP

#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Computations/Activations/ActivationFunc.hpp>

namespace CubbyDNN::Compute
{
class Native
{
public:

    static void Multiply(const Tensor& inputA, const Tensor& inputB,
                         Tensor& output);

    static void Add(const Tensor& inputA, const Tensor& inputB, Tensor& output);

    static void Transpose(const Tensor& input, Tensor& output);

    static void ActivationForward(const Tensor& input, Tensor& output,
                           std::unique_ptr<ActivationFunc>& activation);

    static void ActivationBackward(const Tensor& input, Tensor& output,
                                  std::unique_ptr<ActivationFunc>& activation);

    static void Dot(const Tensor& inputA, const Tensor& inputB, Tensor& output);
};
} // namespace CubbyDNN

#endif
