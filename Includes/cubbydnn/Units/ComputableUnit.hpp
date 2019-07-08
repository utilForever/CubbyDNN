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
#include <vector>

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

/**
 * Wrapper class containing the state and statenum
 */
struct UnitInfo
{
    explicit UnitInfo(const State& state);
    std::atomic<std::size_t> StateNum;
    State CurrentState;
};

class ComputableUnit
{
 public:
    explicit ComputableUnit(State state);

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
    State GetCurrentState()
    {
        return m_unitInfo.CurrentState;
    }

    /**
     * Brings back state number of current unit atomically
     * @return : state number
     */
    std::size_t GetStateNum();

 protected:

    void ChangeState(const State& state);
    /**
     * Atomically increments state number
     */
    void IncrementStateNum(const State& state);

    UnitInfo m_unitInfo;
};

/**
 * Unit that has no input, but has output only.
 * This type of unit must be able to fetch data(from the disk or cache)
 * or generator
 */
class SourceUnit : public ComputableUnit
{
public:
    explicit SourceUnit(const State& state);

    /**
     * Set or add next ComputableUnit ptr
     * @param computableUnitPtr
     */
    void SetNextPtr(SharedPtr<ComputableUnit>&& computableUnitPtr);

    /**
     * Checks if source is ready
     * @return
     */
    bool IsReady() final;

    void Compute() override {
        //DoNothing
        IncrementStateNum(State::ready);
    }

 protected:
    SharedPtr<ComputableUnit> m_nextPtr;
};

/**
 * Unit that has no output, but has inputs
 * This type of unit plays role as sink of the computable graph
 */
class SinkUnit : public ComputableUnit
{
public:
    explicit SinkUnit(const State& state);

    /**
     * Add previous computable Unit to this cell
     * @param computableUnitPtr
     */
    void AddPreviousPtr(SharedPtr<ComputableUnit>&& computableUnitPtr);

    bool IsReady() final;

 protected:
    std::vector<SharedPtr<ComputableUnit>> m_previousPtrVector;
};

class IntermediateUnit : public ComputableUnit
{
 public:
    explicit IntermediateUnit(const State& state);

    void SetNextPtr(SharedPtr<ComputableUnit>&& computableUnitPtr);

    void AddPreviousPtr(SharedPtr<ComputableUnit>&& computableUnitPtr);

 protected:
    std::vector<SharedPtr<ComputableUnit>> m_previousPtr;
    SharedPtr<ComputableUnit> m_nextPtr;
};

/**
 * CopyUnits
 * This will connect between other units(not copy) by copying the output of
 * previous unit to input of next unit
 */
class CopyUnit : ComputableUnit
{
 public:
    explicit CopyUnit(const State& state);

    void SetPreviousPtr(SharedPtr<ComputableUnit>&& computableUnitPtr);

    void SetNextPtr(SharedPtr<ComputableUnit>&& computableUnitPtr);

    void Compute() final{};

 private:
    /// Previous ptr
    SharedPtr<ComputableUnit> m_previousPtr;
    SharedPtr<ComputableUnit> m_nextPtr;
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTABLEUNIT_HPP
