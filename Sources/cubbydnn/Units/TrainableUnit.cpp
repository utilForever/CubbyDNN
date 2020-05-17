/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/TrainableUnit.hpp>

namespace CubbyDNN::Graph
{
TrainableUnit::TrainableUnit(std::vector<Tensor> trainableTensorMap,
                             std::unique_ptr<Computation::Optimizer> optimizer)
    : m_trainableTensorMap(std::move(trainableTensorMap)),
      m_optimizer(std::move(optimizer))
{
}

TrainableUnit& TrainableUnit::operator=(TrainableUnit&& trainableUnit) noexcept
{
    m_trainableTensorMap = std::move(trainableUnit.m_trainableTensorMap);
    m_optimizer = std::move(trainableUnit.m_optimizer);

    return *this;
}

bool TrainableUnit::operator==(const TrainableUnit& trainableUnit) const
{
    return m_trainableTensorMap == trainableUnit.m_trainableTensorMap &&
           m_optimizer == trainableUnit.m_optimizer;
}
}
