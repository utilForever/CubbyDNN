// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_CONSTANTUNIT_HPP
#define TAKION_GRAPH_CONSTANTUNIT_HPP

#include <Takion/Units/SourceComputableUnits/ConstantUnitDecl.hpp>

namespace Takion::Graph
{
template <typename T>
ConstantUnit<T>::ConstantUnit(UnitId unitId, Tensor<T> tensor)
    : ComputableUnit(
          std::move(unitId), tensor.NumericType, {}, {},
          Tensor<T>(tensor.TensorShape, tensor.Device, tensor.NumericType), {}),
      m_value(tensor)
{
}

template <typename T>
ConstantUnit<T>::ConstantUnit(ConstantUnit<T>&& constantUnit) noexcept
    : ComputableUnit<T>(std::move(constantUnit)),
      m_value(std::move(constantUnit.m_value))
{
}

template <typename T>
ConstantUnit ConstantUnit<T>::CreateUnit(const UnitMetaData<T>& unitMetaData)
{
    return ConstantUnit<T>(unitMetaData.Id(),
                        unitMetaData.GetInternalTensor("constant"));
}
} // namespace Takion::Graph

#endif
