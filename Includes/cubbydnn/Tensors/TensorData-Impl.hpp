// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_DATA_IMPL_HPP
#define CUBBYDNN_TENSOR_DATA_IMPL_HPP


#include <cubbydnn/Tensors/TensorData.hpp>

namespace CubbyDNN
{
template <typename T>
TensorData<T>::TensorData(std::vector<T> data, TensorShape shape_)
    : dataVec(std::move(data)), shape(std::move(shape_))
{
    // Do nothing
}
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_DATA_IMPL_HPP