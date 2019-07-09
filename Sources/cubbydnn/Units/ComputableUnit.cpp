// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
UnitInfo::UnitInfo(const State& state) : StateNum(0), CurrentState(state)
{
}

ComputableUnit::ComputableUnit() : m_unitInfo(State::busy)
{
}

void ComputableUnit::ChangeState(const State& state)
{
    m_unitInfo.CurrentState = state;
}

void ComputableUnit::IncrementStateNum()
{
    m_unitInfo.StateNum.fetch_add(1, std::memory_order_release);
}

std::size_t ComputableUnit::GetStateNum()
{
    return m_unitInfo.StateNum;
}

void CopyUnit::SetPreviousPtr(SharedPtr<ComputableUnit>&& computableUnitPtr)
{
    m_previousPtr = std::move(computableUnitPtr);
}

void CopyUnit::SetNextPtr(SharedPtr<ComputableUnit>&& computableUnitPtr)
{
    m_nextPtr = std::move(computableUnitPtr);
}

bool CopyUnit::IsReady()
{
    if (m_unitInfo.CurrentState == State::busy)
        return false;

    auto stateNum = GetStateNum();

    return (m_previousPtr->GetStateNum() == stateNum + 1 &&
            m_nextPtr->GetStateNum() == stateNum);
}

SourceUnit::SourceUnit(TensorInfo outputTensorInfo)
    : m_outputTensorInfo(std::move(outputTensorInfo))
{
}

void SourceUnit::SetNextPtr(SharedPtr<CopyUnit>&& computableUnitPtr)
{
    m_nextPtr = std::move(computableUnitPtr);
}

bool SourceUnit::IsReady()
{
    return m_nextPtr->GetStateNum() == GetStateNum() &&
           m_unitInfo.CurrentState == State::pending;
}

void SinkUnit::AddPreviousPtr(SharedPtr<CopyUnit>&& computableUnitPtr)
{
    m_previousPtrVector.emplace_back(std::move(computableUnitPtr));
}

SinkUnit::SinkUnit(std::vector<TensorInfo> inputTensorInfoVector)
    : m_inputTensorInfoVector(std::move(inputTensorInfoVector))
{
}

bool SinkUnit::IsReady()
{
    if (m_unitInfo.CurrentState == State::busy)
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
      m_outputTensorInfo(std::move(outputTensorInfo))
{
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
    if (m_unitInfo.CurrentState == State::busy)
        return false;

    if (m_nextPtr->GetStateNum() != GetStateNum())

        for (auto& previousPtr : m_previousPtrVector)
        {
            if (previousPtr->GetStateNum() != GetStateNum() + 1)
                return false;
        }
    return true;
}

}  // namespace CubbyDNN