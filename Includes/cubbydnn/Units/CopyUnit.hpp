// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COPYUNIT_HPP
#define CUBBYDNN_COPYUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Utils/WeakPtr.hpp>

namespace CubbyDNN
{
//! CopyUnit
//! This will connect between other units(not copy) by copying the output of
//! previous unit to input of next unit
class CopyUnit
{
public:
    CopyUnit() = default;
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
    void AddOutputPtr(const SharedPtr<ComputableUnit>& computableUnitPtr)
    {
        m_outputUnitPtrVector.emplace_back(computableUnitPtr);
    }

    //! Sets index of previous m_tensor to copy
    //! \param index : index of the previous m_tensor
    void SetInputTensorIndex(size_t index)
    {
        m_inputTensorIndex = index;
    }

    //! Sets index of destination m_tensor to write
    //! \param index : index of the output m_tensor
    void SetOutputTensorIndex(size_t index)
    {
        m_outputTensorIndex = index;
    }

    //! Checks if this copyUnit is ready to be executed
    bool IsReady();

    void ReleaseUnit();

    std::size_t GetStateNum() const;
    //! Implements copy operation between input and output
    void Forward();

    void Backward();

private:
    UnitState m_unitState;

    std::size_t m_inputTensorIndex = 0;
    std::size_t m_outputTensorIndex = 0;

    Tensor m_inputForwardTensor;
    std::vector<Tensor> m_outputForwardTensorVector;

    std::vector<Tensor> m_inputBackwardTensorVector;
    std::vector<Tensor> m_outputBackwardTensorVector;

    /// ptr to source of the copy
    WeakPtr<ComputableUnit> m_inputUnitPtr;
    /// ptr to destination of the copy
    std::vector<WeakPtr<ComputableUnit>> m_outputUnitPtrVector;
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_COPYUNIT_HPP
