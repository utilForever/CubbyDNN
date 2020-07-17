// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include<cubbydnn/Units/SourceComputableUnits/ConstantUnit.hpp>

namespace CubbyDNN::Graph
{
ConstantUnit::ConstantUnit(UnitId unitId, Tensor tensor)
    : ComputableUnit(std::move(unitId), tensor.NumericType, {},
                     {}, std::move(tensor), {})
{
}

ConstantUnit::ConstantUnit(ConstantUnit&& constantUnit) noexcept
    : ComputableUnit(std::move(constantUnit))
{
}

ConstantUnit ConstantUnit::CreateUnit(const UnitMetaData& unitMetaData)
{
    return ConstantUnit(unitMetaData.Id(),
                        unitMetaData.GetInternalTensor("constant"));
}
}
