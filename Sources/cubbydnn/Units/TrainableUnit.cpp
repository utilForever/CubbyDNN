/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/TrainableUnit.hpp>

namespace CubbyDNN::Graph
{
TrainableUnit::TrainableUnit(
    std::unordered_map<std::string, Tensor> trainableTensorMap,
    std::unique_ptr<Compute::Optimizer> optimizer)
    : m_trainableTensorMap(std::move(trainableTensorMap)),
      m_optimizer(std::move(optimizer))
{
}

TrainableUnit::TrainableUnit(
    std::unordered_map<std::string, Tensor> trainableTensorMap)
    : m_trainableTensorMap(std::move(trainableTensorMap))
{
}


TrainableUnit& TrainableUnit::operator=(TrainableUnit&& trainableUnit) noexcept
{
    m_trainableTensorMap = std::move(trainableUnit.m_trainableTensorMap);
    m_optimizer = std::move(trainableUnit.m_optimizer);

    return *this;
}
}
