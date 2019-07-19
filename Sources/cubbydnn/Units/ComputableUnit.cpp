// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
UnitState::UnitState() : StateNum(0), IsBusy(false)
{
}

ComputableUnit::ComputableUnit() : m_unitState()
{
}

std::atomic<std::size_t>& ComputableUnit::GetStateNum()
{
    return m_unitState.StateNum;
}

void CopyUnit::Compute()
{
}

bool CopyUnit::IsReady()
{
    if (ComputableUnit::m_unitState.IsBusy)
        return false;

    auto& stateNum = GetStateNum();
    return (ComputableUnit::m_previousPtrVector.at(0)->GetStateNum() ==
                (stateNum + 1) &&
            m_nextPtr->GetStateNum() == stateNum);
}

SourceUnit::SourceUnit(TensorInfo outputTensorInfo)
    : m_outputTensorInfo(std::move(outputTensorInfo)),
      m_outputTensor(AllocateTensor(outputTensorInfo))
{
}



bool SourceUnit::IsReady()
{
    if (ComputableUnit::m_unitState.IsBusy)
        return false;
    return (m_nextPtr->GetStateNum() == GetStateNum());
}

SinkUnit::SinkUnit(std::vector<TensorInfo> inputTensorInfoVector)
    : m_inputTensorInfoVector(std::move(inputTensorInfoVector))
{
    m_inputTensorVector.reserve(m_inputTensorInfoVector.size());
    for (auto& tensorInfo : m_inputTensorInfoVector)
    {
        m_inputTensorVector.emplace_back(AllocateTensor(tensorInfo));
    }
}

bool SinkUnit::IsReady()
{
    if (ComputableUnit::m_unitState.IsBusy)
        return false;
    for (auto& previousPtr : m_previousPtrVector)
    {
        if (previousPtr->GetStateNum() != GetStateNum() + 1)
            return false;
    }
    return true;
}

IntermediateUnit::IntermediateUnit(
    std::vector<TensorInfo> inputTensorInfoVector, TensorInfo outputTensorInfo)
    : m_inputTensorInfoVector(std::move(inputTensorInfoVector)),
      m_outputTensorInfo(std::move(outputTensorInfo)),
      m_outputTensor(AllocateTensor(outputTensorInfo))
{
    m_inputTensorVector.reserve(m_inputTensorInfoVector.size());

    for (auto& inputTensorInfo : m_inputTensorInfoVector)
    {
        m_inputTensorVector.emplace_back(AllocateTensor(inputTensorInfo));
    }
}

bool IntermediateUnit::IsReady()
{
    if (m_nextPtr->GetStateNum() == GetStateNum())

        for (auto& previousPtr : m_previousPtrVector)
        {
            if (previousPtr->GetStateNum() != GetStateNum() + 1)
                return false;
        }
    return true;
}

}  // namespace CubbyDNN