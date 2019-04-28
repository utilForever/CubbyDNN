// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Tensors/TensorInfo.hpp>

#include <cstddef>
#include <utility>

namespace CubbyDNN
{
TensorInfo::TensorInfo(const TensorShape& tensorShape, bool isMutable)
    : m_isMutable(isMutable), m_shape(tensorShape)
{
}

bool TensorInfo::operator==(const TensorInfo& info) const noexcept
{
    return (m_shape == info.m_shape && m_isMutable == info.m_isMutable);
}

std::size_t TensorInfo::Size()
{
    return m_shape.Size();
}

}  // namespace CubbyDNN