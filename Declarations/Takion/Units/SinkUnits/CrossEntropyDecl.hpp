// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_CROSSENTROPY_DECL_HPP
#define TAKION_GRAPH_CROSSENTROPY_DECL_HPP

#include <Takion/Units/ComputableUnit.hpp>
#include <Takion/FrontEnd/UnitMetaData.hpp>

namespace Takion::Graph
{
template <typename T>
class CrossEntropy : public ComputableUnit<T>
{
public:
    using ComputableUnit<T>::BackwardInputMap;
    using ComputableUnit<T>::BackwardOutputMap;
    using ComputableUnit<T>::ForwardInputMap;
    using ComputableUnit<T>::ForwardOutput;
    using ComputableUnit<T>::m_loss;

    //! \param unitId : subject UnitId
    //! \param predictionUnitId : unitId for prediction
    //! \param labelUnitId : unitId for label
    //! \param predictionTensor : tensor connected to prediction input unit
    //! \param labelTensor : tensor connected to label input unit
    //! \param backwardOutputTensor : tensor that outputs back propagation data
    //! to prediction unit \param batchSize : batch Size
    CrossEntropy(const UnitId& unitId, const UnitId& predictionUnitId,
                 const UnitId& labelUnitId, Tensor<T> predictionTensor,
                 Tensor<T> labelTensor, Tensor<T> backwardOutputTensor,
                 Tensor<T> outputTensor, Compute::Device device,
                 std::size_t batchSize);
    ~CrossEntropy() = default;

    CrossEntropy(const CrossEntropy<T>& lossUnit) = delete;
    CrossEntropy(CrossEntropy<T>&& lossUnit) noexcept;
    CrossEntropy<T>& operator=(const CrossEntropy<T>& lossUnit) = delete;
    CrossEntropy<T>& operator=(CrossEntropy<T>&& lossUnit) noexcept;

    static CrossEntropy<T> CreateUnit(
        const FrontEnd::UnitMetaData<T>& unitMetaData);

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;

private:
    static void m_checkArguments(const Shape& predictionShape,
                                 const Shape& labelShape,
                                 const std::string& unitName);

    UnitId m_predictionUnitId;
    UnitId m_labelUnitId;
    Compute::Device m_device;
};
}

#endif
