// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_DATA_HPP
#define CUBBYDNN_TENSOR_DATA_HPP

#include <cubbydnn/Tensors/TensorInfo.hpp>

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
struct Tensor
{
    Tensor(std::unique_ptr<void> Data, TensorInfo info);
    /// Data vector which possesses actual data
    std::unique_ptr<void> DataPtr;
    /// Shape of this tensorData
    TensorInfo Info;
};


using TensorPtr = typename std::unique_ptr<Tensor>;

/**
 * Builds empty Tensor so data can be put potentially
 * @param info
 * @return
 */
static TensorPtr AllocateTensor(TensorInfo info);

//TODO : Implement methods for Initializing the Tensor

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_DATA_HPP