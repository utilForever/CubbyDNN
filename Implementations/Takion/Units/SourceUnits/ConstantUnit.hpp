// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_CONSTANTUNIT_HPP
#define TAKION_GRAPH_CONSTANTUNIT_HPP

#include <Takion/Units/SourceUnits/ConstantUnitDecl.hpp>

namespace Takion::Graph
{
template <typename T>
ConstantUnit<T>::ConstantUnit(UnitId unitId, Tensor<T> tensor,
                              std::size_t batchSize)
    : ComputableUnit<T>(
        std::move(unitId), {}, {}, tensor, {}, {},
        batchSize)
{
}

template <typename T>
ConstantUnit<T>::ConstantUnit(ConstantUnit<T>&& constantUnit) noexcept
    : ComputableUnit<T>(std::move(constantUnit))
{
}

template <typename T>
ConstantUnit<T> ConstantUnit<T>::CreateUnit(
    const FrontEnd::UnitMetaData<T>& unitMetaData)
{
    const auto& initializer = unitMetaData.GetInitializer("vectorInitializer");
    const auto batchSize = unitMetaData.BatchSize();
    const auto outputShape = unitMetaData.GetOutputShape();
    const auto device = unitMetaData.Device;

    Tensor<T> tensor(outputShape, batchSize, device);
    initializer->Initialize(tensor);

    return ConstantUnit<T>(unitMetaData.Id(), tensor, batchSize);
}
} // namespace Takion::Graph

#endif
