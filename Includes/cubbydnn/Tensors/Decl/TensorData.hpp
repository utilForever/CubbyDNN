// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_DATA_HPP
#define CUBBYDNN_TENSOR_DATA_HPP

#include <cubbydnn/Tensors/TensorShape.hpp>

#include <atomic>
#include <memory>
#include <vector>

namespace CubbyDNN
{
/**
 * TensorData class contains data vector for processing
 * with attributes describing it
 * @tparam T : type of data this tensorData contains
 */
template <typename T>
struct TensorData
{
    TensorData<T>(std::vector<T> data, TensorShape shape);
    /// Data vector which possesses actual data
    // TODO : to void*
    std::vector<T> DataVector;
    /// Shape of this tensorData
    TensorShape Shape;
    /// True if tensorData was set to be mutable
    bool isMutable = true;

    // TensorType Type;
};

template <typename T>
using TensorDataPtr = typename std::unique_ptr<TensorData<T>>;

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_DATA_HPP