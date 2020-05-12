// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/SinkComputableUnits/SinkUnit.hpp>

namespace CubbyDNN::Graph
{
SinkUnit::SinkUnit(UnitId unitId, std::vector<Shape> inputShapeVector,
                   NumberSystem numberSystem)
    : ComputableUnit(unitId, inputShapeVector, Shape(), numberSystem)
{
    ForwardInputVector.reserve(m_inputShapeVector.size());
    for (const auto& tensorShape : m_inputShapeVector)
    {
        ForwardInputVector.emplace_back(
            CreateTensor(tensorShape, numberSystem));
    }
}

SinkUnit::SinkUnit(SinkUnit&& sinkUnit) noexcept
    : ComputableUnit(std::move(sinkUnit))
{
}

SinkUnit& SinkUnit::operator=(SinkUnit&& sinkUnit) noexcept
{
    if (this != &sinkUnit)
        ComputableUnit::operator=(std::move(sinkUnit));
    return *this;
}
} // namespace CubbyDNN
