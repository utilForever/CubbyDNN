// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTABLEUNIT_HPP
#define CUBBYDNN_COMPUTABLEUNIT_HPP

#include <atomic>
#include <cstdlib>
#include <utility>
#include <vector>

#include <cubbydnn/Tensors/TensorInfo.hpp>
#include <cubbydnn/Utils/SharedPtr-impl.hpp>

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
 * Wrapper class containing the state and StateNum
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
    /**
     * Default constructor
     * Initializes unitInfo with pending state
     */
    ComputableUnit();

    /**
     * Constructor
     * @param state : state to initialize unitInfo
     */
    explicit ComputableUnit(State state);

    /**
     * Brings back if executableUnit is ready to be executed
     * @return : whether corresponding unit is ready to be executed
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
    void IncrementStateNum();

    UnitInfo m_unitInfo;
};

/**
 * CopyUnits
 * This will connect between other units(not copy) by copying the output of
 * previous unit to input of next unit
 */
class CopyUnit : public ComputableUnit
{
 public:
    CopyUnit() = default;

    explicit CopyUnit(const State& state);

    void SetPreviousPtr(SharedPtr<ComputableUnit>&& computableUnitPtr);

    void SetNextPtr(SharedPtr<ComputableUnit>&& computableUnitPtr);

    void Compute() final{};

 private:
    /// Previous ptr
    SharedPtr<ComputableUnit> m_previousPtr;
    SharedPtr<ComputableUnit> m_nextPtr;
};

/**
 * Unit that has no input, but has output only.
 * This type of unit must be able to fetch data(from the disk or cache)
 * or generator
 */
class SourceUnit : public ComputableUnit
{
 public:
    /**
     * Constructor
     * @param outputTensorInfo : TensorInfo of the output tensor(Which is always
     * less than 1)
     */
    explicit SourceUnit(TensorInfo outputTensorInfo);

    /**
     * Constructor
     * @param state : state to initialize UnitInfo
     * @param outputTensorInfo : TensorInfo of the output tensor
     */
    SourceUnit(const State& state, TensorInfo outputTensorInfo);

    /**
     * Set or add next ComputableUnit ptr
     * @param computableUnitPtr
     */
    void SetNextPtr(SharedPtr<CopyUnit>&& computableUnitPtr);

    /**
     * Checks if source is ready
     * @return
     */
    bool IsReady() final;

    void Compute() override
    {
        // DoNothing
        IncrementStateNum();
    }

 protected:
    SharedPtr<CopyUnit> m_nextPtr;

    TensorInfo m_outputTensorInfo;
};

/**
 * Unit that has no output, but has inputs
 * This type of unit plays role as sink of the computable graph
 */
class SinkUnit : public ComputableUnit
{
 public:
    explicit SinkUnit(std::vector<TensorInfo> inputTensorInfoVector);

    SinkUnit(const State& state, std::vector<TensorInfo> inputTensorInfoVector);

    /**
     * Add previous computable Unit to this cell
     * @param computableUnitPtr
     */
    void AddPreviousPtr(SharedPtr<CopyUnit>&& computableUnitPtr);

    /**
     * Brings back if executableUnit is ready to be executed
     * @return : whether corresponding unit is ready to be executed
     */
    bool IsReady() final;

 protected:
    std::vector<SharedPtr<CopyUnit>> m_previousPtrVector;

    std::vector<TensorInfo> m_inputTensorInfoVector;
};

class IntermediateUnit : public ComputableUnit
{
 public:
    /**
     * Constructor
     * @param inputTensorInfoVector : vector of TensorInfo
     * @param outputTensorInfo : TensorInfo of the output tensor
     */
    IntermediateUnit(std::vector<TensorInfo> inputTensorInfoVector,
                     TensorInfo outputTensorInfo);

    IntermediateUnit(const State& state,
                     std::vector<TensorInfo> inputTensorInfoVector,
                     TensorInfo outputTensorInfo);

    void SetNextPtr(SharedPtr<CopyUnit>&& computableUnitPtr);

    void AddPreviousPtr(SharedPtr<CopyUnit>&& computableUnitPtr);

 protected:
    std::vector<SharedPtr<CopyUnit>> m_previousPtr;
    SharedPtr<CopyUnit> m_nextPtr;

    std::vector<TensorInfo> m_inputTensorInfoVector;
    TensorInfo m_outputTensorInfo;
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTABLEUNIT_HPP
