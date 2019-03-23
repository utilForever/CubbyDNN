// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_DATA_HPP
#define CUBBYDNN_TENSOR_DATA_HPP

#include <cubbydnn/Tensors/TensorShape.hpp>

#include <vector>

namespace CubbyDNN
{
//!
//! \brief TensorData class.
//!
template <typename T>
class TensorData
{
 public:
    TensorData<T>(std::vector<T> data, TensorShape shape_);

    std::vector<T> dataVec;
    TensorShape shape;
    bool isMutable = true;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_DATA_HPP