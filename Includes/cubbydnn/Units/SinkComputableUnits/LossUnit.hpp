// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_GRAPH_LOSSUNIT_HPP
#define CUBBYDNN_GRAPH_LOSSUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN::Graph
{
class LossUnit : public ComputableUnit
{
public:
    //! \param unitId : subject UnitId
    //! \param predictionUnitId : unitId for prediction
    //! \param labelUnitId : unitId for label
    //! \param predictionTensor : tensor connected to prediction input unit
    //! \param labelTensor : tensor connected to label input unit
    //! \param backwardOutputTensor : tensor that outputs back propagation data to prediction unit
    //! \param lossType : Type of loss function to use
    //! \param numberSystem : number system to use
    LossUnit(const UnitId& unitId, const UnitId& predictionUnitId,
             const UnitId& labelUnitId,
             Tensor predictionTensor, Tensor labelTensor,
             Tensor backwardOutputTensor, std::string lossType,
             NumberSystem numberSystem);
    ~LossUnit() = default;

    LossUnit(const LossUnit& lossUnit) = delete;
    LossUnit(LossUnit&& lossUnit) noexcept;
    LossUnit& operator=(const LossUnit& lossUnit) = delete;
    LossUnit& operator=(LossUnit&& lossUnit) noexcept;

    static LossUnit CreateUnit(const UnitMetaData& unitMetaData);

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
