// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_LOSSUNIT_HPP
#define TAKION_GRAPH_LOSSUNIT_HPP

#include <Takion/Units/ComputableUnit.hpp>

namespace Takion::Graph
{
template <typename T>
class LossUnit : public ComputableUnit<T>
{
public:
    //! \param unitId : subject UnitId
    //! \param predictionUnitId : unitId for prediction
    //! \param labelUnitId : unitId for label
    //! \param predictionTensor : tensor connected to prediction input unit
    //! \param labelTensor : tensor connected to label input unit
    //! \param backwardOutputTensor : tensor that outputs back propagation data to prediction unit
    //! \param lossType : Type of loss function to use
    //! \param batchSize : batch Size
    LossUnit(const UnitId& unitId, const UnitId& predictionUnitId,
             const UnitId& labelUnitId,
             Tensor predictionTensor, Tensor labelTensor,
             Tensor backwardOutputTensor, std::string lossType,
             std::size_t batchSize);
    ~LossUnit() = default;

    LossUnit(const LossUnit<T>& lossUnit) = delete;
    LossUnit(LossUnit<T>&& lossUnit) noexcept;
    LossUnit<T>& operator=(const LossUnit<T>& lossUnit) = delete;
    LossUnit<T>& operator=(LossUnit<T>&& lossUnit) noexcept;

    static LossUnit<T> CreateUnit(const UnitMetaData<T>& unitMetaData);

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;

private:
    std::string m_lossType;
    UnitId m_predictionUnitId;
    UnitId m_labelUnitId;
};
}

#endif
