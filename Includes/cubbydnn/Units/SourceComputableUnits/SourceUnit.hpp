/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SOURCEUNIT_HPP
#define CUBBYDNN_SOURCEUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
//! Unit that has no input, but has output only.
//! This type of unit must be able to fetch data(from the disk or cache)
//! or generator
class SourceUnit : public ComputableUnit
{
public:
    //! Constructor
    //! \param output : TensorInfo of the output
    //! \param numberOfOutputs : number of connections to this source unit
    explicit SourceUnit(TensorInfo output, size_t numberOfOutputs = 1);
    ~SourceUnit() = default;

    //! SourceUnit is not copy-assignable
    SourceUnit(const SourceUnit& sourceUnit) = delete;
    SourceUnit(SourceUnit&& sourceUnit) noexcept;

    //! SourceUnit is not copy-assignable
    SourceUnit& operator=(const SourceUnit& sourceUnit) = delete;
    SourceUnit& operator=(SourceUnit&& sourceUnit) noexcept;

    //! Checks if source is ready
    //! \return : true if ready to be computed false otherwise
    bool IsReady() final;

    void Compute() override
    {
    }
};

class ConstantUnit : public SourceUnit
{
public:
    explicit ConstantUnit(TensorInfo output, int numberOfOutputs,
                          void* dataPtr);
    ~ConstantUnit();

    //! ConstantUnit is not copy-assignable
    ConstantUnit(const ConstantUnit& constantUnit) = delete;
    ConstantUnit(ConstantUnit&& constantUnit) noexcept;

    //! ConstantUnit is not copy-assignable
    ConstantUnit& operator=(const ConstantUnit& sourceUnit) = delete;
    ConstantUnit& operator=(ConstantUnit&& constantUnit) noexcept;

private:
    void* m_dataPtr = nullptr;
    size_t m_byteSize = 0;
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_SOURCEUNIT_HPP
