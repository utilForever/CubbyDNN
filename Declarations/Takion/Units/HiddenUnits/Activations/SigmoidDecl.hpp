// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_SIGMOID_DECL_HPP
#define TAKION_GRAPH_SIGMOID_DECL_HPP

#include <Takion/FrontEnd/UnitMetaData.hpp>
#include <Takion/Units/ComputableUnit.hpp>

namespace Takion::Graph
{
template <typename T>
class Sigmoid : public ComputableUnit<T>
{
public:
    using ComputableUnit<T>::BackwardInputMap;
    using ComputableUnit<T>::BackwardOutputMap;
    using ComputableUnit<T>::ForwardInputMap;
    using ComputableUnit<T>::ForwardOutput;
    using ComputableUnit<T>::InternalTensorMap;

    Sigmoid(const UnitId& unitId, UnitId sourceUnitId, Tensor<T> forwardInput,
            std::unordered_map<UnitId, Tensor<T>> backwardInputVector,
            Tensor<T> forwardOutput, Tensor<T> backwardOutput,
            std::unordered_map<std::string, Tensor<T>> internalTensorMap,
            std::size_t batchSize);
    ~Sigmoid() = default;

    Sigmoid(const Sigmoid& activationUnit) = delete;
    Sigmoid(Sigmoid&& activationUnit) noexcept;
    Sigmoid& operator=(const Sigmoid& activationUnit) = delete;
    Sigmoid& operator=(Sigmoid&& activationUnit) noexcept;

    static Sigmoid<T> CreateUnit(const FrontEnd::UnitMetaData<T>& unitMetaData);

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;

    void ChangeBatchSize(std::size_t batchSize) override;

private:
    UnitId m_sourceUnitId;

    static void m_checkArguments(const Shape& inputShape,
                                 const Shape& outputShape,
                                 const std::string& unitName);
};
} // namespace Takion::Graph

#endif
