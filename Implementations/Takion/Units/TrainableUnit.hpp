// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_TRAINABLEUNIT_HPP
#define TAKION_GRAPH_TRAINABLEUNIT_HPP

#include <Takion/Units/TrainableUnitDecl.hpp>

namespace Takion::Graph
{
template <typename T>
TrainableUnit<T>::TrainableUnit(
    std::unordered_map<std::string, Tensor<T>> trainableTensorMap,
    std::unique_ptr<Compute::Optimizer<T>> optimizer)
    : TrainableTensorMap(std::move(trainableTensorMap)),
      m_optimizer(std::move(optimizer))
{
}

template <typename T>
TrainableUnit<T>::TrainableUnit(
    std::unordered_map<std::string, Tensor<T>> trainableTensorMap)
    : TrainableTensorMap(std::move(trainableTensorMap))
{
}

template <typename T>
TrainableUnit<T>& TrainableUnit<T>::operator=(TrainableUnit<T>&& trainableUnit)
noexcept
{
    TrainableTensorMap = std::move(trainableUnit.TrainableTensorMap);
    m_optimizer = std::move(trainableUnit.m_optimizer);

    return *this;
}
} // namespace Takion::Graph

#endif
