// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SINKUNIT_HPP
#define CUBBYDNN_SINKUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace Takion::Graph
{
//! Unit that has no output, but has inputs
//! This type of unit plays role as sink of the computable graph
class SinkUnit : public ComputableUnit
{
public:
    //! \param unitId : Unique Id of this unit
    //! \param numberSystem : number system for this unit
    //! \param forwardInputVector :vector of input tensor for forward propagation
    //! \param backwardOutputVector : output of backward propagation
    explicit SinkUnit(
        UnitId unitId,
        NumberSystem numberSystem,
        std::vector<Tensor> forwardInputVector,
        std::vector<Tensor> backwardOutputVector);
    virtual ~SinkUnit() = default;

    //! SinkUnit is not copy-assignable
    SinkUnit(const SinkUnit& sinkUnit) = delete;
    SinkUnit(SinkUnit&& sinkUnit) noexcept;

    //! SinkUnit is not copy-assignable
    SinkUnit& operator=(const SinkUnit& sinkUnit) = delete;
    SinkUnit& operator=(SinkUnit&& sinkUnit) noexcept;
};
} // namespace Takion

#endif  // CUBBYDNN_SINKUNIT_HPP
