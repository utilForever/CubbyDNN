// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
UnitState::UnitState() = default;

ComputableUnit::ComputableUnit(UnitType unitType)
    : Type(unitType)
{
}

ComputableUnit::ComputableUnit(ComputableUnit&& other) noexcept
    : Type(other.Type)
{
}

ComputableUnit& ComputableUnit::operator=(ComputableUnit&& other) noexcept
{
    if (this == &other)
        return *this;
    Type = other.Type;
    return *this;
}


void ComputableUnit::ReleaseUnit()
{
    m_unitState.StateNum.fetch_add(1, std::memory_order_release);
}
} // namespace CubbyDNN
