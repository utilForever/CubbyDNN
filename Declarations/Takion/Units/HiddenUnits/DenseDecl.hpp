// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_DENSE_DECL_HPP
#define TAKION_GRAPH_DENSE_DECL_HPP

#include <Takion/Units/ComputableUnit.hpp>
#include <Takion/FrontEnd/UnitMetaData.hpp>
#include <Takion/Units/TrainableUnit.hpp>

namespace Takion::Graph
{
template <typename T>
class DenseUnit : public ComputableUnit<T>, public TrainableUnit<T>
{
public:
    using ComputableUnit<T>::ForwardInputMap;
    using TrainableUnit<T>::TrainableTensorMap;
    using ComputableUnit<T>::ForwardOutput;
    using ComputableUnit<T>::BackwardInputMap;
    using ComputableUnit<T>::BackwardOutputMap;
    using ComputableUnit<T>::InternalTensorMap;
    using TrainableUnit<T>::m_optimizer;


    DenseUnit(const UnitId& unitId, const UnitId& sourceUnitId,
              Tensor<T> forwardInput,
              std::unordered_map<UnitId, Tensor<T>> backwardInputMap,
              Tensor<T> forwardOutput, Tensor<T> backwardOutput,
              std::unordered_map<std::string, Tensor<T>> internalTensorMap,
              std::unordered_map<std::string, Tensor<T>> trainableTensorMap,
              std::unique_ptr<Compute::Optimizer<T>> optimizer,
              std::size_t batchSize);
    ~DenseUnit() = default;

    DenseUnit(const DenseUnit<T>& denseUnit) = delete;
    DenseUnit(DenseUnit<T>&& denseUnit) noexcept;
    DenseUnit& operator=(const DenseUnit<T>& denseUnit) = delete;
    DenseUnit& operator=(DenseUnit<T>&& denseUnit) noexcept;

    static DenseUnit<T> CreateUnit(
        const FrontEnd::UnitMetaData<T>& unitMetaData,
        std::unique_ptr<Compute::Optimizer<T>> optimizer);

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;

    void ChangeBatchSize(std::size_t batchSize) override;

private:
    UnitId m_sourceUnitId;
    static void m_checkShape(const Shape& inputShape, const Shape& outputShape,
                             const Shape& weightShape, const Shape& biasShape,
                             const std::string& unitName);
};
} // namespace Takion

#endif
