// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_ACTIVATIONUNIT_DECL_HPP
#define TAKION_GRAPH_ACTIVATIONUNIT_DECL_HPP

#include <Takion/Units/ComputableUnit.hpp>
#include <Takion/Units/TrainableUnit.hpp>
#include <Takion/Units/UnitMetaData.hpp>


namespace Takion::Graph
{
template <typename T>
class ReLU
    : public ComputableUnit<T>,
      public TrainableUnit<T>
{
public:
    ReLU(const UnitId& unitId, const UnitId& sourceUnitId,
                   Tensor<T> forwardInput,
                   std::unordered_map<UnitId, Tensor<T>> backwardInputVector,
                   Tensor<T> forwardOutput, Tensor<T> backwardOutput,
                   std::unordered_map<std::string, Tensor<T>> trainableUnit);
    ~ReLU() = default;

    ReLU(const ReLU& activationUnit) = delete;
    ReLU(ReLU&& activationUnit) noexcept;
    ReLU& operator=(const ReLU& activationUnit) = delete;
    ReLU& operator=(ReLU&& activationUnit) noexcept;

    static ReLU CreateUnit(const UnitMetaData& unitMetaData);

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;

private:
    UnitId m_sourceUnitId;
};
} // namespace Takion::Graph

#endif
