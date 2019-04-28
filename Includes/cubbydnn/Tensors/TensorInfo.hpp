// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_INFO_HPP
#define CUBBYDNN_TENSOR_INFO_HPP
#include <cubbydnn/Tensors/TensorShape.hpp>
namespace CubbyDNN
{
//!
//! \brief TensorInfo class.
//!
class TensorInfo
{
 public:
    explicit TensorInfo(const TensorShape& tensorShape, bool isMutable = false);

    bool operator==(const TensorInfo& info) const noexcept;

    std::size_t Size();

 private:
    bool m_isMutable;
    const TensorShape& m_shape;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_INFO_HPP