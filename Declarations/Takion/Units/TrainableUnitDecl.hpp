// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_TRAINABLEUNIT_DECL_HPP
#define TAKION_GRAPH_TRAINABLEUNIT_DECL_HPP

#include <memory>
#include <unordered_map>
#include <Takion/Tensors/Tensor.hpp>
#include <Takion/Computations/Optimizers/Optimizer.hpp>

namespace Takion::Graph
{
template <typename T>
class TrainableUnit

{
public:
    TrainableUnit(std::unordered_map<std::string, Tensor<T>> trainableTensorMap,
                  std::unique_ptr<Compute::Optimizer<T>> optimizer);

    TrainableUnit(
        std::unordered_map<std::string, Tensor<T>> trainableTensorMap);

    virtual ~TrainableUnit() = default;

    TrainableUnit(const TrainableUnit<T>& trainableUnit) = delete;

    TrainableUnit(TrainableUnit<T>&& trainableUnit) noexcept
        : TrainableTensorMap(std::move(trainableUnit.TrainableTensorMap)),
          m_optimizer(std::move(trainableUnit.m_optimizer))
    {
    }

    TrainableUnit<T>& operator=(const TrainableUnit<T>& trainableUnit) = delete;
    TrainableUnit<T>& operator=(TrainableUnit<T>&& trainableUnit) noexcept;

    std::unordered_map<std::string, Tensor<T>> TrainableTensorMap;

protected:
    std::unique_ptr<Compute::Optimizer<T>> m_optimizer = nullptr;
};
}

#endif
