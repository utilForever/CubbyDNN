// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/UnitId.hpp>

namespace CubbyDNN::Graph
{
UnitType::UnitType(UnitBaseType type, std::string_view name)
    : BaseType(type),
      m_typeName(name)
{
}

UnitType::UnitType(UnitBaseType type, std::string_view name,
               SharedPtr<UnitType> baseUnit)
    : BaseUnit(std::move(baseUnit)),
      BaseType(type),
      m_typeName(name)
{
}

bool UnitType::operator==(const UnitType& rhs) const
{
    return BaseType == rhs.BaseType &&
           m_typeName == rhs.m_typeName;
}

bool UnitType::operator!=(const UnitType& rhs) const
{
    return !(*this == rhs);
}

bool UnitType::IsBaseOf(const UnitType& baseUnit, const UnitType& derivedUnit)
{
    if (!derivedUnit.BaseUnit.Get())
    {
        return false;
    }

    if (*derivedUnit.BaseUnit.Get() == baseUnit)
        return true;

    return IsBaseOf(baseUnit, *derivedUnit.BaseUnit.Get());
}
}
