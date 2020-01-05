//
// Created by jwkim98 on 8/13/19.
//

#ifndef CUBBYDNN_SOURCEUNIT_HPP
#define CUBBYDNN_SOURCEUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Units/CopyUnit.hpp>

namespace CubbyDNN
{
//! Unit that has no input, but has output only.
//! This type of unit must be able to fetch data(from the disk or cache)
//! or generator
class SourceUnit : public ComputableUnit
{
 public:
    //! Constructor
    //! \param outputTensorInfoVector : TensorInfo of the output m_tensor(Which is
    //! always less than 1)
    explicit SourceUnit(std::vector<TensorInfo> outputTensorInfoVector);

    SourceUnit(SourceUnit&& sourceUnit) noexcept;

    //! Checks if source is ready
    //! \return : true if ready to be computed false otherwise
    bool IsReady() final;

    void Compute() override
    {
       // std::cout << "SourceUnit" << std::endl;
       // std::cout << m_unitState.StateNum << std::endl;
    }

};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_SOURCEUNIT_HPP
