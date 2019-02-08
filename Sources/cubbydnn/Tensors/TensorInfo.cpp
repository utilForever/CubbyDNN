// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Tensors/TensorInfo.hpp>

#include <utility>

namespace CubbyDNN
{
TensorInfo::TensorInfo(long from, long to, bool) : m_from(from), m_to(to)
{
    // Do nothing
}

bool TensorInfo::operator==(const TensorInfo& info) const
{
    return (m_from == info.m_from && m_to == info.m_to);
}

unsigned TensorInfo::ProcessCount() const
{
    return m_processCount;
}

void TensorInfo::IncrementProcessCount()
{
    m_processCount += 1;
}
}  // namespace CubbyDNN