// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_LOSSUNIT_DECL_HPP
#define TAKION_GRAPH_LOSSUNIT_DECL_HPP

#include <Takion/Units/ComputableUnit.hpp>

namespace Takion::Graph
{
template <typename T>
class MSELoss : public ComputableUnit<T>
{
public:
    //! \param unitId : subject UnitId
    //! \param predictionUnitId : unitId for prediction
    //! \param labelUnitId : unitId for label
    //! \param predictionTensor : tensor connected to prediction input unit
    //! \param labelTensor : tensor connected to label input unit
    //! \param backwardOutputTensor : tensor that outputs back propagation data to prediction unit
    //! \param batchSize : batch Size
    MSELoss(const UnitId& unitId, const UnitId& predictionUnitId,
            const UnitId& labelUnitId,
            Tensor<T> predictionTensor, Tensor<T> labelTensor,
            Tensor<T> backwardOutputTensor,
            std::size_t batchSize);
    ~MSELoss() = default;

    MSELoss(const MSELoss<T>& lossUnit) = delete;
    MSELoss(MSELoss<T>&& lossUnit) noexcept;
    MSELoss<T>& operator=(const MSELoss<T>& lossUnit) = delete;
    MSELoss<T>& operator=(MSELoss<T>&& lossUnit) noexcept;

    static MSELoss<T> CreateUnit(const UnitMetaData<T>& unitMetaData);

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;

private:
    UnitId m_predictionUnitId;
    UnitId m_labelUnitId;
};
}

#endif
