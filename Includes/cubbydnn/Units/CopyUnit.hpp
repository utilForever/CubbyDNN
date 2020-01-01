// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COPYUNIT_HPP
#define CUBBYDNN_COPYUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Utils/WeakPtr-impl.hpp>

namespace CubbyDNN
{
//! CopyUnit
//! This will connect between other units(not copy) by copying the output of
//! previous unit to input of next unit
class CopyUnit : public ComputableUnit
{
public:
    CopyUnit();

    ~CopyUnit() = default;

    CopyUnit(CopyUnit& copyUnit) = delete;

    CopyUnit(CopyUnit&& copyUnit) noexcept;

    CopyUnit& operator=(CopyUnit& copyUnit) = delete;

    CopyUnit& operator=(CopyUnit&& copyUnit) = delete;

    //! Sets ComputableUnitPtr of previous unit to copy from
    //! If no computableUnitPtr has been assigned, unit is added. If it has
    //! previously assigned computableUnitPtr, computableUnitPtr is replaced by
    //! given parameter
    //! \param computableUnitPtr : computableUnitPtr to add or replace
    void SetInputPtr(const SharedPtr<ComputableUnit>& computableUnitPtr)
    {
        m_inputUnitPtr = computableUnitPtr;
    }

    //! Sets ComputableUnitPtr of previous unit to copy from
    //! If no computableUnitPtr has been assigned, unit is added. If it has
    //! previously assigned computableUnitPtr, computableUnitPtr is replaced by
    //! given parameter
    //! \param computableUnitPtr : computableUnitPtr to add or replace
    void SetOutputPtr(const SharedPtr<ComputableUnit>& computableUnitPtr)
    {
        m_outputUnitPtr = computableUnitPtr;
    }


    //! Sets index of previous tensor to copy
    //! \param index : index of the previous tensor
    void SetInputTensorIndex(size_t index)
    {
        m_inputTensorIndex = index;
    }

    //! Sets index of destination tensor to write
    //! \param index : index of the output tensor
    void SetOutputTensorIndex(size_t index)
    {
        m_outputTensorIndex = index;
    }

    //! Checks if this copyUnit is ready to be executed
    bool IsReady() override;

    //! Implements copy operation between input and output
    void Compute() override;


private:

    size_t m_inputTensorIndex = 0;
    size_t m_outputTensorIndex = 0;

    /// ptr to source of the copy
    WeakPtr<ComputableUnit> m_inputUnitPtr;
    /// ptr to destination of the copy
    WeakPtr<ComputableUnit> m_outputUnitPtr;
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_COPYUNIT_HPP
