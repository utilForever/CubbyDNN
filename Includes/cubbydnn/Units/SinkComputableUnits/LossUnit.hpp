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
    LossUnit(UnitId unitId, NumberSystem numberSystem, Tensor forwardInput,
             Tensor label, Tensor delta, std::string lossName);
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
    std::string m_lossName;
};
}

#endif
