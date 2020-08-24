// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_ABSTENSOR_DECL_HPP
#define TAKION_ABSTENSOR_DECL_HPP

#include <Takion/Utils/Shape.hpp>
#include <Takion/Units/UnitType.hpp>

namespace Takion::FrontEnd
{
template <typename T>
class AbsTensor
{
public:
    AbsTensor(Shape shape, UnitId sourceId)
        : m_shape(shape),
          m_sourceUnitId(sourceId)
    {
    }

    [[nodiscard]] UnitId GetPrevOutput() const
    {
        return m_sourceUnitId;
    }

    [[nodiscard]] Shape GetShape() const
    {
        return m_shape;
    }

    friend bool operator<(const AbsTensor& lhs, const AbsTensor& rhs)
    {
        return lhs.m_sourceUnitId < rhs.m_sourceUnitId;
    }

    friend bool operator<=(const AbsTensor& lhs, const AbsTensor& rhs)
    {
        return !(rhs < lhs);
    }

    friend bool operator>(const AbsTensor& lhs, const AbsTensor& rhs)
    {
        return rhs < lhs;
    }

    friend bool operator>=(const AbsTensor& lhs, const AbsTensor& rhs)
    {
        return !(lhs < rhs);
    }

private:
    Shape m_shape;
    UnitId m_sourceUnitId;
};
}

#endif
