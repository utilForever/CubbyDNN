// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_DATA_HPP
#define CUBBYDNN_TENSOR_DATA_HPP

#include <cubbydnn/Tensors/TensorInfo.hpp>

#include <atomic>
#include <cassert>
#include <cstring>
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
    Tensor(void* Data, TensorInfo info);

    ~Tensor()
    {
        free(DataPtr);
    }

    Tensor(Tensor&& tensor) noexcept;
    /// move assignment operator
    Tensor& operator=(Tensor&& tensor) noexcept;
    /// Data vector which possesses actual data
    void* DataPtr;
    /// Shape of this tensorData
    TensorInfo Info;
};

void CopyTensor(Tensor& source, Tensor& destination);

/**
 * Builds empty Tensor so data can be put potentially
 * @param info
 * @return
 */
Tensor AllocateTensor(const TensorInfo& info);

// TODO : Implement methods for Initializing the Tensor

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_DATA_HPP