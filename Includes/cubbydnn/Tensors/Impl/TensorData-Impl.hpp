/** Copyright (c) 2019 Chris Ohk, Justin Kim
 *
* We are making my contributions/submissions to this project solely in our
* personal capacity and are not conveying any rights to any intellectual
* property of any third parties.
*/

#ifndef CUBBYDNN_TENSOR_DATA_IMPL_HPP
#define CUBBYDNN_TENSOR_DATA_IMPL_HPP


#include <cubbydnn/Tensors/Decl/TensorData.hpp>

namespace CubbyDNN
{
template <typename T>
TensorData<T>::TensorData(std::vector<T> data, TensorShape shape)
    : DataVector(std::move(data)), Shape(std::move(shape))
{
    // Do nothing
}
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_DATA_IMPL_HPP