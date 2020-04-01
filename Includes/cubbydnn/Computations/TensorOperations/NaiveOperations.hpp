// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_NAIVEOPERATIONS_HPP
#define CUBBYDNN_NAIVEOPERATIONS_HPP

#include <cubbydnn/Tensors/Tensor.hpp>

namespace CubbyDNN
{
class Native
{
public:
    static void Multiply(const Tensor& inputA, const Tensor& inputB,
                         Tensor& output);

    static void Add(const Tensor& inputA, const Tensor& inputB, Tensor& output);

    static void Transpose(const Tensor& input, Tensor& output);

    static void Activation(const Tensor& input, Tensor& output)
    {
        //TODO : implement activation functions in this way
        Tensor::CopyTensor(input, output);
        throw std::runtime_error("Not implemented");
    }
};
} // namespace CubbyDNN

#endif
