// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_MATRIX_HPP
#define CUBBYDNN_MATRIX_HPP
#include <cubbydnn/Tensors/Tensor.hpp>

namespace CubbyDNN
{
void IdentityMatrix(const Shape& shape, NumberSystem numberSystem);

void Multiply(const Tensor& inputA, const Tensor& inputB,
                     Tensor& output);
void Add();

void Transpose();

void dot();
}

#endif
