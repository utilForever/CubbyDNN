//
// Created by jwkim98 on 8/13/19.
//

#ifndef CUBBYDNN_COPYUNIT_HPP
#define CUBBYDNN_COPYUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
/**
 * CopyUnits
 * This will connect between other units(not copy) by copying the output of
 * previous unit to input of next unit
 */
class CopyUnit : public ComputableUnit
{
 public:
    CopyUnit();

    CopyUnit(CopyUnit&& copyUnit) noexcept;

    //! Sets ComputableUnitPtr of previous unit to copy from
    //! If no computableUnitPtr has been assigned, unit is added. If it has
    //! previously assigned computableUnitPtr, computableUnitPtr is replaced by
    //! given parameter
    //! \param computableUnitPtr : computableUnitPtr to add or replace
    void SetInputPtr(ComputableUnit* computableUnitPtr)
    {
        ComputableUnit::m_inputPtrVector.at(m_inputIndex++) = computableUnitPtr;
    }

    //! Sets ComputableUnitPtr of previous unit to copy from
    //! If no computableUnitPtr has been assigned, unit is added. If it has
    //! previously assigned computableUnitPtr, computableUnitPtr is replaced by
    //! given parameter
    //! \param computableUnitPtr : computableUnitPtr to add or replace
    void SetOutputPtr(ComputableUnit* computableUnitPtr)
    {
        ComputableUnit::m_outputPtrVector.at(m_outputIndex++) =
            computableUnitPtr;
    }

    Tensor& GetInputTensor(size_t index) override
    {
        index;
        assert(false);
        return tensor;
    }

    Tensor& GetOutputTensor(size_t index) override
    {
        index;
        assert(false);
        return tensor;
    }

    void SetInputTensorIndex(size_t index)
    {
        m_inputTensorIndex = index;
    }

    void SetOutputTensorIndex(size_t index)
    {
        m_outputTensorIndex = index;
    }

    //! Implements copy operation between input and output
    void Compute() override;
    //! Checks if this copyUnit is ready to be executed
    bool IsReady() override;

 private:
    size_t m_inputTensorIndex;
    size_t m_outputTensorIndex;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_COPYUNIT_HPP
