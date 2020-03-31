// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SINKUNIT_HPP
#define CUBBYDNN_SINKUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
//! Unit that has no output, but has inputs
//! This type of unit plays role as sink of the computable graph
class SinkUnit : public ComputableUnit
{
public:
    //! Constructor
    //! \param unitId : Unique Id of this unit
    //! \param inputShapeVector : vector of tensorInfo to accept
    //! \param numberSystem : number system for this unit
    explicit SinkUnit(UnitId unitId, std::vector<Shape> inputShapeVector,
                      NumberSystem numberSystem);
    ~SinkUnit() = default;

    //! SinkUnit is not copy-assignable
    SinkUnit(const SinkUnit& sinkUnit) = delete;
    SinkUnit(SinkUnit&& sinkUnit) noexcept;

    //! SinkUnit is not copy-assignable
    SinkUnit& operator=(const SinkUnit& sinkUnit) = delete;
    SinkUnit& operator=(SinkUnit&& sinkUnit) noexcept;
};

class CrossEntropy : public ComputableUnit
{
public:
    explicit CrossEntropy(UnitId unitId, Shape inputShape,
                          NumberSystem numberSystem)
        : ComputableUnit(unitId, { std::move(inputShape) }, Shape(),
                         numberSystem)
    {
    };

    void Forward() override
    {
    };

    void Backward() override
    {
    };
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_SINKUNIT_HPP
