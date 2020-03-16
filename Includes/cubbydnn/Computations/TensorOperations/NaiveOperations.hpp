// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_NAIVEOPERATIONS_HPP
#define CUBBYDNN_NAIVEOPERATIONS_HPP

#include <cubbydnn/Computations/TensorOperations/TensorOperations.hpp>
#include <cubbydnn/Tensors/Tensor.hpp>

namespace CubbyDNN
{
class NaiveOperation : public TensorOperation
{
 public:
    NaiveOperation();

    void Multiply(const Tensor& inputA, const Tensor& inputB,
                  Tensor& output) override;

    void Add(const Tensor& inputA, const Tensor& inputB,
             Tensor& output) override;

    void Transpose(const Tensor& input, Tensor& output) override;

    void Activation(const Tensor& input, Tensor& output) override
    {
        input;
        output;
         throw std::runtime_error("Not implemented");
    }
};
}  // namespace CubbyDNN

#endif
