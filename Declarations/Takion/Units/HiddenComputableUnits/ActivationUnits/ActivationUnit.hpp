// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_GRAPH_ACTIVATIONUNIT_HPP
#define CUBBYDNN_GRAPH_ACTIVATIONUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Units/TrainableUnit.hpp>
#include <cubbydnn/Units/UnitMetadata.hpp>


namespace Takion::Graph
{
class ActivationUnit : public ComputableUnit, public TrainableUnit
{
public:
    ActivationUnit(const UnitId& unitId, const UnitId& sourceUnitId,
                   Tensor forwardInput,
                   std::unordered_map<UnitId, Tensor> backwardInputVector,
                   Tensor forwardOutput, Tensor backwardOutput,
                   std::unordered_map<std::string, Tensor> trainableUnit,
                   std::string activationType, NumberSystem numberSystem);
    ~ActivationUnit() = default;

    ActivationUnit(const ActivationUnit& activationUnit) = delete;
    ActivationUnit(ActivationUnit&& activationUnit) noexcept;
    ActivationUnit& operator=(const ActivationUnit& activationUnit) = delete;
    ActivationUnit& operator=(ActivationUnit&& activationUnit) noexcept;

    static ActivationUnit CreateUnit(const UnitMetaData& unitMetaData);

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;

private:
    std::string m_activationType;
    UnitId m_sourceUnitId;
};
} // namespace Takion::Graph

#endif
