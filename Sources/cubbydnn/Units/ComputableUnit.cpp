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

ComputableUnit::ComputableUnit(State state) : m_unitInfo(state)
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

CopyUnit::CopyUnit(const State& state) : ComputableUnit(state)
{
}

void CopyUnit::SetPreviousPtr(SharedPtr<ComputableUnit>&& computableUnitPtr)
{
    m_previousPtr = std::move(computableUnitPtr);
}

void CopyUnit::SetNextPtr(SharedPtr<ComputableUnit>&& computableUnitPtr)
{
    m_nextPtr = std::move(computableUnitPtr);
}

SourceUnit::SourceUnit(TensorInfo outputTensorInfo)
    : m_outputTensorInfo(std::move(outputTensorInfo))
{
}

SourceUnit::SourceUnit(const State& state, TensorInfo outputTensorInfo)
    : ComputableUnit(state), m_outputTensorInfo(std::move(outputTensorInfo))
{
}

void SourceUnit::SetNextPtr(SharedPtr<CopyUnit>&& computableUnitPtr)
{
    m_nextPtr = std::move(computableUnitPtr);
}

bool SourceUnit::IsReady()
{
    return m_nextPtr->GetStateNum() == GetStateNum();
}

void SinkUnit::AddPreviousPtr(SharedPtr<CopyUnit>&& computableUnitPtr)
{
    m_previousPtrVector.emplace_back(std::move(computableUnitPtr));
}

SinkUnit::SinkUnit(std::vector<TensorInfo> inputTensorInfoVector)
    : m_inputTensorInfoVector(std::move(inputTensorInfoVector))
{
}

SinkUnit::SinkUnit(const State& state,
                   std::vector<TensorInfo> inputTensorInfoVector)
    : ComputableUnit(state),
      m_inputTensorInfoVector(std::move(inputTensorInfoVector))
{
}

bool SinkUnit::IsReady()
{
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

IntermediateUnit::IntermediateUnit(
    const State& state, std::vector<TensorInfo> inputTensorInfoVector,
    TensorInfo outputTensorInfo)
    : ComputableUnit(state),
      m_inputTensorInfoVector(std::move(inputTensorInfoVector)),
      m_outputTensorInfo(std::move(outputTensorInfo))
{
}

void IntermediateUnit::SetNextPtr(SharedPtr<CopyUnit>&& computableUnitPtr)
{
    m_nextPtr = std::move(computableUnitPtr);
}

void IntermediateUnit::AddPreviousPtr(SharedPtr<CopyUnit>&& computableUnitPtr)
{
    m_previousPtr.emplace_back(std::move(computableUnitPtr));
}

}  // namespace CubbyDNN