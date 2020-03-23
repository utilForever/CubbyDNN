/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SOURCEUNIT_HPP
#define CUBBYDNN_SOURCEUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Utils/SharedPtr.hpp>

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
    explicit SourceUnit(TensorInfo output);
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

    std::size_t AddOutputPtr(
        const SharedPtr<ComputableUnit>& computableUnitPtr);

    virtual void Forward()
    {
    }

    virtual void Backward()
    {
        //! No default action required for back propagation for source unit
    }

 protected:
    /// ptr to units to write result
    std::vector<SharedPtr<ComputableUnit>> m_outputPtrVector;
};

class PlaceHolderUnit : public SourceUnit
{
    explicit PlaceHolderUnit(TensorInfo shape);
};

class ConstantUnit : public SourceUnit
{
 public:
    explicit ConstantUnit(TensorInfo output, void* dataPtr);
    ~ConstantUnit();

    //! ConstantUnit is not copy-assignable
    ConstantUnit(const ConstantUnit& constantUnit) = delete;
    ConstantUnit(ConstantUnit&& constantUnit) noexcept;

    //! ConstantUnit is not copy-assignable
    ConstantUnit& operator=(const ConstantUnit& sourceUnit) = delete;
    ConstantUnit& operator=(ConstantUnit&& constantUnit) noexcept;

    void Forward() final;

 private:
    void* m_dataPtr = nullptr;
    std::size_t m_byteSize = 0;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_SOURCEUNIT_HPP
