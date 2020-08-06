// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_DENSE_DECL_HPP
#define TAKION_GRAPH_DENSE_DECL_HPP

#include <Takion/Units/ComputableUnit.hpp>
#include <Takion/Units/UnitMetaData.hpp>
#include <Takion/Units/TrainableUnit.hpp>

namespace Takion::Graph
{
template <typename T>
class DenseUnit : public ComputableUnit<T>, public TrainableUnit<T>
{
public:
    DenseUnit(const UnitId& unitId, const UnitId& sourceUnitId,
              Tensor<T> forwardInput,
              std::unordered_map<UnitId, Tensor<T>> backwardInputMap,
              Tensor<T> forwardOutput, Tensor<T> backwardOutput,
              std::unordered_map<std::string, Tensor<T>> trainableUnit,
              std::unique_ptr<Compute::Optimizer<T>> optimizer);
    ~DenseUnit() = default;

    DenseUnit(const DenseUnit& denseUnit) = delete;
    DenseUnit(DenseUnit&& denseUnit) noexcept;
    DenseUnit& operator=(const DenseUnit& denseUnit) = delete;
    DenseUnit& operator=(DenseUnit&& denseUnit) noexcept;

    static DenseUnit CreateUnit(
        const UnitMetaData& unitMetaData,
        std::unique_ptr<Compute::Optimizer<T>> optimizer);

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;

private:
    UnitId m_sourceUnitId;
};
} // namespace Takion

#endif
