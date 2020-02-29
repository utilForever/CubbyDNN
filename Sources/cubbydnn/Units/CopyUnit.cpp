// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/CopyUnit.hpp>

namespace CubbyDNN
{
CopyUnit::CopyUnit() : ComputableUnit(UnitType::Copy)
{
}

void CopyUnit::Compute()
{
    auto& inputTensor = m_inputUnitPtr->GetOutputTensor(m_inputTensorIndex);

    auto& outputTensor = m_outputUnitPtr->GetInputTensor(m_outputTensorIndex);
    CopyTensor(inputTensor, outputTensor);
}

bool CopyUnit::IsReady()
{
    if (m_unitState.IsBusy.load(std::memory_order_acquire))
        return false;

    const bool ready =  m_inputUnitPtr->GetStateNum() ==
               m_unitState.StateNum.load(std::memory_order_acquire) + 1 &&
           m_outputUnitPtr->GetStateNum() ==
               m_unitState.StateNum.load(std::memory_order_acquire);
    return ready;
}
}  // namespace CubbyDNN
