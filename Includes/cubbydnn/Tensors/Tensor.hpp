// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_DATA_HPP
#define CUBBYDNN_TENSOR_DATA_HPP

#include <cubbydnn/Tensors/TensorDataInfo.hpp>

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
struct Tensor'Data
{
    TensorData(void* Data, NumberSystem numberSystem, const TensorDataInfo& info);
    /// Data vector which possesses actual data
    void* DataPtr;
    /// Type of this ptr
    NumberSystem numberSystem;
    /// Shape of this tensorData
    TensorDataInfo Info;
    // TensorType Type;
};



using TensorDataPtr = typename std::unique_ptr<TensorData>;

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_DATA_HPP