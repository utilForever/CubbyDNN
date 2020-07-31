// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_DENSE_HPP
#define CUBBYDNN_DENSE_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Units/UnitMetadata.hpp>
#include <cubbydnn/Units/TrainableUnit.hpp>

namespace Takion::Graph
{
class DenseUnit : public ComputableUnit, public TrainableUnit
{
public:
    DenseUnit(const UnitId& unitId, const UnitId& sourceUnitId,
              Tensor forwardInput,
              std::unordered_map<UnitId, Tensor> backwardInputMap,
              Tensor forwardOutput, Tensor backwardOutput,
              std::unordered_map<std::string, Tensor> trainableUnit,
              std::unique_ptr<Compute::Optimizer> optimizer,
              NumberSystem numberSystem);
    ~DenseUnit() = default;

    DenseUnit(const DenseUnit& denseUnit) = delete;
    DenseUnit(DenseUnit&& denseUnit) noexcept;
    DenseUnit& operator=(const DenseUnit& denseUnit) = delete;
    DenseUnit& operator=(DenseUnit&& denseUnit) noexcept;

    static DenseUnit CreateUnit(
        const UnitMetaData& unitMetaData,
        std::unique_ptr<Compute::Optimizer> optimizer);

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;

private:
    UnitId m_sourceUnitId;
};
} // namespace Takion

#endif
