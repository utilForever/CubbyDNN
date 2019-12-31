// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/CopyUnit.hpp>

namespace CubbyDNN
{
CopyUnit::CopyUnit() : ComputableUnit(1, 1, UnitType::Copy)
{
}

CopyUnit::CopyUnit(CopyUnit&& copyUnit) noexcept
    : ComputableUnit(std::move(copyUnit)),
      m_inputTensorIndex(copyUnit.m_inputTensorIndex),
      m_outputTensorIndex(copyUnit.m_outputTensorIndex),
      m_inputUnitPtr(std::move(copyUnit.m_inputUnitPtr)),
      m_outputUnitPtr(std::move(copyUnit.m_outputUnitPtr))
{
}

void CopyUnit::Compute()
{
    std::cout << "CopyUnit" << std::endl;
    std::cout << m_unitState.StateNum << std::endl;

    auto& inputTensor = m_inputUnitPtr->GetOutputTensor(m_inputTensorIndex);

    auto& outputTensor = m_outputUnitPtr->GetInputTensor(m_outputTensorIndex);
    CopyTensor(inputTensor, outputTensor);
}

bool CopyUnit::IsReady()
{
    if (m_unitState.IsBusy.load(std::memory_order_seq_cst))
        return false;

    auto stateNum = m_outputUnitPtr->GetStateNum();
    const bool ready =  m_inputUnitPtr->GetStateNum() ==
               m_unitState.StateNum.load(std::memory_order_seq_cst) + 1 &&
           m_outputUnitPtr->GetStateNum() ==
               m_unitState.StateNum.load(std::memory_order_seq_cst);
    stateNum;
    return ready;
}
}  // namespace CubbyDNN
