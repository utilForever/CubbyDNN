/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TRAINABLEUNIT_HPP
#define CUBBYDNN_TRAINABLEUNIT_HPP

#include <memory>
#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Computations/Optimizers/Optimizer.hpp>

namespace CubbyDNN::Graph
{
class TrainableUnit
{
public:
    TrainableUnit(std::vector<Tensor> trainableTensorMap,
                  std::unique_ptr<Computation::Optimizer> optimizer);
    virtual ~TrainableUnit() = default;

    TrainableUnit(const TrainableUnit& trainableUnit) = delete;
    TrainableUnit(TrainableUnit&& trainableUnit) noexcept = default;
    TrainableUnit& operator=(const TrainableUnit& trainableUnit) = delete;
    TrainableUnit& operator=(TrainableUnit&& trainableUnit) noexcept;

protected:
    std::vector<Tensor> m_trainableTensorMap;
   std::unique_ptr<Computation::Optimizer> m_optimizer;
};
}

#endif
