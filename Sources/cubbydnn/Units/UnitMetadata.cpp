// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/UnitMetadata.hpp>

namespace CubbyDNN::Graph
{
UnitMetaData::UnitMetaData(UnitId unitId, std::vector<Shape> inputShapeVector,
                           Shape outputShape,
                           std::vector<UnitId> inputUnitIdVector,
                           std::vector<UnitId> outputUnitIdVector,
                           std::vector<Initializer> initializerVector,
                           NumberSystem numericType, std::size_t padSize)
    : NumericType(numericType),
      PadSize(padSize),
      m_unitId(std::move(unitId)),
      m_inputShapeVector(std::move(inputShapeVector)),
      m_outputShape(std::move(outputShape)),
      m_inputUnitVector(std::move(inputUnitIdVector)),
      m_outputUnitVector(std::move(outputUnitIdVector)),
      m_initializerVector(std::move(initializerVector))
{
}

UnitId UnitMetaData::Id() const
{
    return m_unitId;
}

std::vector<Shape> UnitMetaData::InputShapeVector() const
{
    return m_inputShapeVector;
}

Shape UnitMetaData::OutputShape() const
{
    return m_outputShape;
}

std::vector<UnitId> UnitMetaData::InputUnitVector() const
{
    return m_inputUnitVector;
}

std::vector<UnitId> UnitMetaData::OutputUnitVector() const
{
    return m_outputUnitVector;
}
}
