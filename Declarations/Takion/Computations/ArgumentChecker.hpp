// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_ARGUMENTCHECKER_HPP
#define TAKION_COMPUTE_ARGUMENTCHECKER_HPP
#include <Takion/Tensors/TensorDecl.hpp>

namespace Takion::Compute
{
template <typename T>
void CheckMultiplyArguments(const Tensor<T>& A, const Tensor<T>& B,
                            Tensor<T>& out)
{
    const auto shapeA = A.TensorShape;
    const auto shapeB = B.TensorShape;
    const auto shapeOut = out.TensorShape;

    if (shapeA.NumRow() != shapeOut.NumRow())
        throw std::invalid_argument(
            "Multiply - Shape mismatch inputA NumRow(" + shapeA.NumRow() +
            ") while output NumRow(" + shapeOut.NumRow() + ")");

    if (shapeB.NumCol() != shapeOut.NumCol())
        throw std::invalid_argument(
            "Multiply - Shape mismatch - inputB NumCol(" +
            shapeB.NumCol() + ") while output NumCol(" +
            shapeOut.NumCol() + ")");

    if (shapeA.NumCol() != shapeB.NumRow())
        throw std::invalid_argument(
            "Multiply - Shape mismatch - inputA NumCol(" +
            shapeA.NumCol() + ") while input B NumRow(" +
            shapeB.NumRow() + ")");

}


}

#endif
