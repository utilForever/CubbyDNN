// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Computations/ComputableUnit.hpp>

namespace CubbyDNN
{
UnitInfo::UnitInfo(const State& state) : StateNum(0), CurrentState(state)
{
}

ComputableUnit::ComputableUnit(State state) : m_unitState(state)
{
}

void ComputableUnit::IncrementState(const State& state)
{
    m_unitState.CurrentState = state;
    m_unitState.StateNum.fetch_add(1, std::memory_order_release);
}

std::size_t ComputableUnit::GetStateNum()
{
    return m_unitState.StateNum;
}

SourceUnit::SourceUnit(const State& state) : ComputableUnit(state)
{
}

void SourceUnit::SetNextPtr(SharedPtr<ComputableUnit>&& computableUnitPtr)
{
    m_nextPtr = std::move(computableUnitPtr);
}

bool SourceUnit::IsReady()
{
    return m_nextPtr->GetStateNum() == GetStateNum();
}

void SinkUnit::AddPreviousPtr(SharedPtr<ComputableUnit>&& computableUnitPtr)
{
    m_previousPtrVector.emplace_back(std::move(computableUnitPtr));
}

SinkUnit::SinkUnit(const State& state) : ComputableUnit(state)
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

IntermediateUnit::IntermediateUnit(const State& state) : ComputableUnit(state)
{
}

void IntermediateUnit::SetNextPtr(SharedPtr<ComputableUnit>&& computableUnitPtr)
{
    m_nextPtr = std::move(computableUnitPtr);
}

void IntermediateUnit::AddPreviousPtr(
    SharedPtr<ComputableUnit>&& computableUnitPtr)
{
    m_previousPtr.emplace_back(std::move(computableUnitPtr));
}

void Copy::SetPreviousPtr(SharedPtr<ComputableUnit>&& computableUnitPtr)
{
    m_previousPtr = std::move(computableUnitPtr);
}

void Copy::SetNextPtr(SharedPtr<ComputableUnit>&& computableUnitPtr)
{
    m_nextPtr = std::move(computableUnitPtr);
}

}  // namespace CubbyDNN