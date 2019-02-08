// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_DATA_HPP
#define CUBBYDNN_TENSOR_DATA_HPP

#include <cubbydnn/Tensor/TensorShape.hpp>

#include <vector>

namespace CubbyDNN
{
//!
//! \brief TensorData class.
//!
class TensorData
{
 public:
    TensorData(std::vector<float> data, TensorShape shape_);

    std::vector<float> dataVec;
    TensorShape shape;
    bool isMutable = true;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_DATA_HPP