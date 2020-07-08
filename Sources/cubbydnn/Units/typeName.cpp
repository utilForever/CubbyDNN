// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/typeName.hpp>

namespace CubbyDNN::Graph
{
typeName::typeName(UnitBaseType type, std::string_view name)
    : BaseType(type),
      m_typeName(name)
{
}

typeName::typeName(UnitBaseType type, std::string_view name,
               SharedPtr<typeName> baseUnit)
    : BaseUnit(std::move(baseUnit)),
      BaseType(type),
      m_typeName(name)
{
}

bool typeName::operator==(const typeName& unitType) const
{
    return BaseType == unitType.BaseType &&
           m_typeName == unitType.m_typeName;
}

bool typeName::operator!=(const typeName& unitType) const
{
    return !(*this == unitType);
}

bool typeName::IsBaseOf(const typeName& baseUnit, const typeName& derivedUnit)
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
