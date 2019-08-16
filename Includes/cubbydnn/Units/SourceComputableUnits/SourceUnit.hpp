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
    //! \param outputTensorInfoVector : TensorInfo of the output tensor(Which is
    //! always less than 1)
    explicit SourceUnit(std::vector<TensorInfo> outputTensorInfoVector);

    SourceUnit(SourceUnit&& sourceUnit) noexcept;

    //! Set or add next ComputableUnit ptr
    //! \param computableUnitPtr : computablePtr to set
    size_t AddOutputPtr(CopyUnit* computableUnitPtr)
    {
        ComputableUnit::m_outputPtrVector.at(m_outputIndex) = computableUnitPtr;
        return m_outputIndex++;
    }

    //! Checks if source is ready
    //! \return : true if ready to be computed false otherwise
    bool IsReady() final;

    void Compute() override
    {
        std::cout << "SourceUnit" << std::endl;
        std::cout << m_unitState.StateNum << std::endl;
    }

    Tensor& GetInputTensor(size_t index) override
    {
        return tensor;
    }

    Tensor& GetOutputTensor(size_t index) override
    {
        return m_outputTensorVector.at(index);
    }

 private:
    std::vector<TensorInfo> m_outputTensorInfoVector;
    std::vector<Tensor> m_outputTensorVector;
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_SOURCEUNIT_HPP
