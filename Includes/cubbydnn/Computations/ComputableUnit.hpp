// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTABLEUNIT_HPP
#define CUBBYDNN_COMPUTABLEUNIT_HPP

#include <cubbydnn/Utils/SharedPtr-impl.hpp>

#include <atomic>
#include <cstdlib>
#include <utility>

namespace CubbyDNN
{
/**
 * Describes state of the unit
 */
enum class State
{
    pending,
    ready,
    busy,
};
struct UnitState
{
    explicit UnitState(const State& state);
    std::atomic<std::size_t> StateNum;
    State CurrentState;
};

class ComputableUnit
{
 public:
    explicit ComputableUnit(State state);
    /**
     * Starts executableUnit by enqueueing into the engine
     */
    virtual void Start() = 0;
    /**
     * Finishes operation by sending end signal
     */
    virtual void Finish() = 0;

    /**
     * Brings back if executableUnit is ready to be executed
     * @return
     */
    virtual bool IsReady() = 0;

    /**
     * Method that is executed on the engine
     */
    virtual void Compute() = 0;

    /**
     * Brings back state of the executableUnit
     * @return : state of the operation
     */
    State GetState()
    {
        return m_unitState.CurrentState;
    }

    /**
     * Atomically increments state number
     */
    void UpdateState(const State& state);

    /**
     * Brings back state number of current unit atomically
     * @return
     */
    std::size_t GetStateNum();

 protected:
    UnitState m_unitState;
};

class Copy : ComputableUnit
{
 public:
    Copy();

    void SetPreviousPtr(SharedPtr<ComputableUnit>&& computableUnitPtr);

    void SetNextPtr(SharedPtr<ComputableUnit>&& computableUnitPtr);

 private:
    SharedPtr<ComputableUnit> m_previousPtr;
    SharedPtr<ComputableUnit> m_nextPtr;
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTABLEUNIT_HPP
