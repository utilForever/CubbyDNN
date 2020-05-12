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
    //! Constructor
    //! \param output : TensorInfo of the output
    explicit SourceUnit(UnitId unitId, Shape outputShape,
                        NumberSystem numberSystem);
    ~SourceUnit() = default;

    //! SourceUnit is not copy-assignable
    SourceUnit(const SourceUnit& sourceUnit) = delete;
    SourceUnit(SourceUnit&& sourceUnit) noexcept;

    //! SourceUnit is not copy-assignable
    SourceUnit& operator=(const SourceUnit& sourceUnit) = delete;
    SourceUnit& operator=(SourceUnit&& sourceUnit) noexcept;
};

class PlaceHolderUnit : public ComputableUnit
{
public:
    explicit PlaceHolderUnit(UnitId unitId, Shape outputShape,
                             NumberSystem numberSystem);
    ~PlaceHolderUnit() = default;

    PlaceHolderUnit(const PlaceHolderUnit& placeHolderUnit) = delete;
    PlaceHolderUnit(PlaceHolderUnit&& placeHolderUnit) noexcept;

    PlaceHolderUnit& operator=(const PlaceHolderUnit& placeHolderUnit) = delete;
    PlaceHolderUnit& operator=(PlaceHolderUnit&& placeHolderUnit) noexcept;

    void Forward() override
    {
    };

    void Backward() override
    {
    };
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_SOURCEUNIT_HPP
