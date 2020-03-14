// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_INITIALIZER_HPP
#define CUBBYDNN_INITIALIZER_HPP

#include <cubbydnn/Computations/Initializers/InitializerOp.hpp>
#include <cubbydnn/Tensors/Tensor.hpp>

namespace CubbyDNN
{
class Initializer
{
    static void RandomNormal(Tensor& tensor, double mean, double stddev);

    static void RandomUniform(Tensor& tensor, double min, double max);

    static void XavierNormal(Tensor& tensor);

    static void LecunNormal(Tensor& tensor);

    static void HeNormal(Tensor& tensor);

    static void XavierUniform(Tensor& tensorInfo);

    static void LecunUniform(Tensor& tensorInfo);

    static void HeUniform(Tensor& tensorInfo);

   static  void Zeros(Tensor& tensor);

    static void Ones(Tensor& tensor);
};
}

#endif
