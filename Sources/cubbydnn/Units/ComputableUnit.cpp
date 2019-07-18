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
    if(ComputableUnit::m_unitState.IsBusy)
        return false;

    auto& stateNum = GetStateNum();
    return (m_previousPtr->GetStateNum() == (stateNum + 1) &&
            m_nextPtr->GetStateNum() == stateNum);
}

SourceUnit::SourceUnit(TensorInfo outputTensorInfo)
    : m_outputTensorInfo(std::move(outputTensorInfo)),
      m_outputTensor(AllocateTensor(outputTensorInfo))
{
}

void SourceUnit::SetNextPtr(SharedPtr<CopyUnit>&& computableUnitPtr)
{
    m_nextPtr = std::move(computableUnitPtr);
}

bool SourceUnit::IsReady()
{
    if(ComputableUnit::m_unitState.IsBusy)
        return false;
    return (m_nextPtr->GetStateNum()==GetStateNum());
}

void SinkUnit::AddPreviousPtr(SharedPtr<CopyUnit>&& computableUnitPtr)
{
    m_previousPtrVector.emplace_back(std::move(computableUnitPtr));
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
    if(ComputableUnit::m_unitState.IsBusy)
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

void IntermediateUnit::SetNextPtr(SharedPtr<CopyUnit>&& computableUnitPtr)
{
    m_nextPtr = std::move(computableUnitPtr);
}

void IntermediateUnit::AddPreviousPtr(SharedPtr<CopyUnit>&& computableUnitPtr)
{
    m_previousPtrVector.emplace_back(std::move(computableUnitPtr));
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