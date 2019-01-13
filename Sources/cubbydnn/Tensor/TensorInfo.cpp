// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Tensor/TensorInfo.hpp>

#include <utility>

namespace CubbyDNN
{
TensorInfo::TensorInfo(long from, long to, bool) : m_from(from), m_to(to)
{
    // Do nothing
}
}  // namespace CubbyDNN