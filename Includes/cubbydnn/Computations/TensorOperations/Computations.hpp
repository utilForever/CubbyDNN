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
void MultiplyAdd(const Tensor& inputA, const Tensor& inputB,
                 const Tensor& inputC, Tensor& output, bool transposeA,
                 bool transposeB);
//
// void BatchMultiply(const Tensor& inputA, const Tensor& batchedInputB,
//                    Tensor& output,
//                    bool transposeA = false, bool transposeB = false);

//! Broadcasts inputA to inputB
//! \param inputA : input to be broadcasted
//! \param inputB : input to receive broadcasted operation
//! \param output : output With 
void BroadcastMultiply(const Tensor& inputA, const Tensor& inputB,
                       Tensor& output, std::size_t dim);

void Multiply(const Tensor& inputA,const Tensor& inputB,
              Tensor& output, bool transposeA = false, bool transposeB = false,
              bool broadCast = false);

void Add(const Tensor& inputA, const Tensor& inputB, Tensor& output,
         bool broadCast = false);

void Add(Tensor& tensor, const Tensor& toAdd, bool broadCast = false);

//! Adds up all tensors in tensorVector and outputs result to output
// void Add(const std::vector<Tensor>& tensorVector, Tensor& output);

//! Returns mean of input tensor from axis
//! This function assumes the highest dimension is for batch
//! \param idx : index that indicates end of batch
void Shrink(Tensor& tensor, Tensor& output, int index = -1);

void Transpose(const Tensor& input, Tensor& output);

void Dot(const Tensor& inputA, const Tensor& inputB, Tensor& output,
         bool broadcast = false);

void ScalarMul(const Tensor& input, Tensor& output, float toMul);

void ScalarMul(Tensor& tensor, float toMul);
} // namespace CubbyDNN
#endif
