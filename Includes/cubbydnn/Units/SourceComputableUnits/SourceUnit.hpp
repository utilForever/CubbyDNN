/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SOURCEUNIT_HPP
#define CUBBYDNN_SOURCEUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN::Graph
{
//! Unit that has no input, but has output only.
//! This type of unit must be able to fetch data(from the disk or cache)
//! or generator
class SourceUnit : public ComputableUnit
{
public:
    //! \param unitId : id of the unit
    //! \param numberSystem : number system to use
    //! propagation
    //! \param backwardInputVector : vector of input tensor for back
    //! propagation
    //! \param forwardOutput : output of forward propagation
    explicit SourceUnit(UnitId unitId,
                        NumberSystem numberSystem, Tensor forwardOutput,
                        std::vector<Tensor> backwardInputVector);
    ~SourceUnit() = default;

    //! SourceUnit is not copy-assignable
    SourceUnit(const SourceUnit& sourceUnit) = delete;
    SourceUnit(SourceUnit&& sourceUnit) noexcept;

    //! SourceUnit is not copy-assignable
    SourceUnit& operator=(const SourceUnit& sourceUnit) = delete;
    SourceUnit& operator=(SourceUnit&& sourceUnit) noexcept;
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_SOURCEUNIT_HPP
