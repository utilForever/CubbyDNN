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
    //! \param outputTensorInfoVector : TensorInfo of the output m_tensor(Which
    //! is always less than 1)
    explicit SourceUnit(std::vector<TensorInfo> outputTensorInfoVector);

    //! SourceUnit is not copy-assignable
    SourceUnit(const SourceUnit& sourceUnit) = delete;

    ~SourceUnit() = default;

    //! SourceUnit is not copy-assignable
    SourceUnit& operator=(const SourceUnit& sourceUnit) = delete;

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
    explicit ConstantUnit(TensorInfo output, int numberOfOutputs, void* dataPtr);

    //! ConstantUnit is not copy-assignable
    ConstantUnit(const ConstantUnit& sourceUnit) = delete;

    ~ConstantUnit()
    {
        free(m_dataPtr);
    }

    //! ConstantUnit is not copy-assignable
    SourceUnit& operator=(const SourceUnit& sourceUnit) = delete;
private:
   void* m_dataPtr = nullptr;
   size_t m_byteSize = 0;
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_SOURCEUNIT_HPP
