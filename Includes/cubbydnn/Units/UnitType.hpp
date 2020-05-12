// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_UNITID_HPP
#define CUBBYDNN_UNITID_HPP
#include <cubbydnn/Utils/SharedPtr.hpp>
#include <string_view>

namespace CubbyDNN::Graph
{
enum class UnitBaseType
{
    Source,
    Hidden,
    Sink,
    Copy,
};

class UnitType
{
public:
    UnitType(UnitBaseType type, std::string_view name);
    UnitType(UnitBaseType type, std::string_view name,
             SharedPtr<UnitType> baseUnit);
    ~UnitType() = default;

    UnitType(const UnitType& unitType) = default;
    UnitType(UnitType&& unitType) noexcept = default;
    UnitType& operator=(const UnitType& unitType) = default;
    UnitType& operator=(UnitType&& unitType) noexcept = default;

    bool operator==(const UnitType& rhs) const;
    bool operator!=(const UnitType& rhs) const;

    [[nodiscard]] bool IsBaseOf(const UnitType& derivedUnit) const
    {
        return IsBaseOf(*this, derivedUnit);
    }

    [[nodiscard]] bool IsDerivedFrom(const UnitType& baseUnit) const
    {
        return IsBaseOf(baseUnit, *this);
    }

    static bool IsBaseOf(const UnitType& baseUnit, const UnitType& derivedUnit);

    SharedPtr<UnitType> BaseUnit;
    UnitBaseType BaseType;

private:
    std::string m_typeName;
};

struct UnitId
{
    UnitType Type;
    std::size_t Id;
    std::string UnitName;
};
;
} // namespace CubbyDNN

#endif
