// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_SOFTMAX_DECL_HPP
#define TAKION_GRAPH_SOFTMAX_DECL_HPP

#include <Takion/Units/ComputableUnit.hpp>
#include <Takion/FrontEnd/UnitMetaData.hpp>


namespace Takion::Graph
{
template <typename T>
class SoftMax : public ComputableUnit<T>
{
public:
    using ComputableUnit<T>::BackwardInputMap;
    using ComputableUnit<T>::BackwardOutputMap;
    using ComputableUnit<T>::ForwardInputMap;
    using ComputableUnit<T>::ForwardOutput;
    using ComputableUnit<T>::InternalTensorMap;

    SoftMax(const UnitId& unitId, UnitId sourceUnitId, Tensor<T> forwardInput,
            std::unordered_map<UnitId, Tensor<T>> backwardInputVector,
            Tensor<T> forwardOutput, Tensor<T> backwardOutput,
            std::unordered_map<std::string, Tensor<T>> internalTensorMap,
            Compute::Device device,
            std::size_t batchSize);

    ~SoftMax() = default;

    SoftMax(const SoftMax& softMax) = delete;
    SoftMax(SoftMax&& softMax) noexcept = default;
    SoftMax& operator=(const SoftMax& softMax) = delete;
    SoftMax& operator=(SoftMax&& softMax) noexcept = default;

    static SoftMax<T> CreateUnit(const FrontEnd::UnitMetaData<T>& unitMetaData);

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;

    void ChangeBatchSize(std::size_t batchSize) override;

private:

    static void m_checkArguments(const Shape& inputShape,
                                 const Shape& outputShape,
                                 const std::string& unitName);

    UnitId m_sourceUnitId;
    Compute::Device m_device;
};
}


#endif
