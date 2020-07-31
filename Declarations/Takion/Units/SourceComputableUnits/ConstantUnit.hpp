// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_GRAPH_CONSTANTUNIT_HPP
#define CUBBYDNN_GRAPH_CONSTANTUNIT_HPP

#include <cubbydnn/Units/SourceComputableUnits/SourceUnit.hpp>

namespace Takion::Graph
{
class ConstantUnit : public ComputableUnit
{
public:
    ConstantUnit(UnitId unitId, Tensor tensor);
    ~ConstantUnit() override = default;

    ConstantUnit(const ConstantUnit& constantUnit) = delete;
    ConstantUnit(ConstantUnit&& constantUnit) noexcept;

    ConstantUnit& operator=(const ConstantUnit& constantUnit) = delete;
    ConstantUnit& operator=(ConstantUnit&& constantUnit) noexcept = default;

    static ConstantUnit CreateUnit(const UnitMetaData& unitMetaData);

    void Forward() override
    {
        Tensor::CopyTensorData(m_value, ForwardOutput);
    }

    void AsyncForward(std::promise<bool> promise) override
    {
        Tensor::CopyTensorData(m_value, ForwardOutput);
        promise.set_value(true);
    }

    void Backward() override
    {
    }

    void AsyncBackward(std::promise<bool> promise) override
    {
        promise.set_value(true);
    }

private:
    Tensor m_value;
};
}

#endif
