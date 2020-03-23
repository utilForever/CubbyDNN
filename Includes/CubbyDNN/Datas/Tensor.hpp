#ifndef CUBBYDNN_TENSOR_HPP
#define CUBBYDNN_TENSOR_HPP

#include <vector>

namespace CubbyDNN
{
template <typename DType>
using Tensor = std::vector<DType>;

using BoolTensor = Tensor<bool>;
using LongTensor = Tensor<long>;
using FloatTensor = Tensor<float>;
}

#endif  // CUBBYDNN_TENSOR_HPP
