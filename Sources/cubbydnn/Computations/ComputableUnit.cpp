// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Computations/ComputableUnit.hpp>

namespace CubbyDNN
{
UnitState::UnitState(const State& state) : StateNum(0), CurrentState(state)
{
}

ComputableUnit::ComputableUnit(State state) : m_unitState(state)
{
}

void ComputableUnit::UpdateState(const State& state)
{
    m_unitState.CurrentState = state;
    m_unitState.StateNum.fetch_add(1, std::memory_order_release);
}

std::size_t ComputableUnit::GetStateNum()
{
    return m_unitState.StateNum;
}

Copy::Copy() : ComputableUnit(State::pending)
{
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
