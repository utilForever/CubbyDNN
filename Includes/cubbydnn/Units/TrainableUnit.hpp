/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TRAINABLEUNIT_HPP
#define CUBBYDNN_TRAINABLEUNIT_HPP

#include <memory>
#include <unordered_map>
#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Computations/Optimizers/Optimizer.hpp>

namespace CubbyDNN::Graph
{
class TrainableUnit
{
public:
    TrainableUnit(std::unordered_map<std::string, Tensor> trainableTensorMap,
                  std::unique_ptr<Compute::Optimizer> optimizer);

    TrainableUnit(std::unordered_map<std::string, Tensor> trainableTensorMap);

    virtual ~TrainableUnit() = default;

    TrainableUnit(const TrainableUnit& trainableUnit) = delete;

    TrainableUnit(TrainableUnit&& trainableUnit) noexcept
        : m_trainableTensorMap(std::move(trainableUnit.m_trainableTensorMap)),
          m_optimizer(std::move(trainableUnit.m_optimizer))
    {
    }

    TrainableUnit& operator=(const TrainableUnit& trainableUnit) = delete;
    TrainableUnit& operator=(TrainableUnit&& trainableUnit) noexcept;

protected:
    std::unordered_map<std::string, Tensor> m_trainableTensorMap;
    std::unique_ptr<Compute::Optimizer> m_optimizer = nullptr;
};
}

#endif
