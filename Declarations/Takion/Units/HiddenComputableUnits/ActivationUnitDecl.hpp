// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_ACTIVATIONUNIT_DECL_HPP
#define TAKION_GRAPH_ACTIVATIONUNIT_DECL_HPP

#include <Takion/Units/ComputableUnit.hpp>
#include <Takion/Units/TrainableUnit.hpp>
#include <Takion/Units/UnitMetadata.hpp>


namespace Takion::Graph
{
template <typename T>
class ActivationUnit
    : public ComputableUnit<T>,
      public TrainableUnit<T>
{
public:
    ActivationUnit(const UnitId& unitId, const UnitId& sourceUnitId,
                   Tensor<T> forwardInput,
                   std::unordered_map<UnitId, Tensor<T>> backwardInputVector,
                   Tensor<T> forwardOutput, Tensor<T> backwardOutput,
                   std::unordered_map<std::string, Tensor<T>> trainableUnit,
                   std::string activationType);
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
