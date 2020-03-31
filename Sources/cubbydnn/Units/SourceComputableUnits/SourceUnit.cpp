// Copyright (c) 2019 ChrisOhk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cassert>
#include <cubbydnn/Units/SourceComputableUnits/SourceUnit.hpp>

namespace CubbyDNN
{
SourceUnit::SourceUnit(UnitId unitId, Shape outputShape,
                       NumberSystem numberSystem)
    : ComputableUnit(unitId, {}, std::move(outputShape), numberSystem)
{
}

SourceUnit::SourceUnit(SourceUnit&& sourceUnit) noexcept
    : ComputableUnit(std::move(sourceUnit))
{
}

SourceUnit& SourceUnit::operator=(SourceUnit&& sourceUnit) noexcept
{
    if (this == &sourceUnit)
        return *this;
    ComputableUnit::operator=(std::move(sourceUnit));
    return *this;
}

PlaceHolderUnit::PlaceHolderUnit(UnitId unitId, Shape outputShape,
                                 NumberSystem numberSystem)
    : ComputableUnit(unitId, {}, outputShape, numberSystem)
{
}

PlaceHolderUnit::PlaceHolderUnit(PlaceHolderUnit&& placeHolderUnit) noexcept
    : ComputableUnit(std::move(placeHolderUnit))
{
}

PlaceHolderUnit& PlaceHolderUnit::operator=(
    PlaceHolderUnit&& placeHolderUnit) noexcept
{
    if (this == &placeHolderUnit)
        return *this;
    ComputableUnit::operator=(std::move(placeHolderUnit));
    return *this;
}
} // namespace CubbyDNN
