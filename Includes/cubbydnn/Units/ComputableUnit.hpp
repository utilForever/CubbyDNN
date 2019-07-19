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

#include <cubbydnn/Tensors/Tensor.hpp>
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
    busy,
};

/**
 * UnitState
 * Wrapper class containing the state and StateNum
 * this represents the execution state of computable unit
 */
struct UnitState
{
    explicit UnitState();
    /// State number of current
    std::atomic<std::size_t> StateNum;
    /// True if unit is already in the task queue
    std::atomic<bool> IsBusy;
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
     * Called before computation for acquiring the unit in order to compute
     * Marks IsBusy as True
     */
    void AcquireUnit()
    {
        setBusy();
    }

    /**
     * Called after computation for releasing the unit after computation
     * Increments the stateNum and marks IsBusy as false
     */
    void ReleaseUnit()
    {
        incrementStateNum();
        setReleased();
    }

    /**
     * Brings back if executableUnit is ready to be executed
     * @return : whether corresponding unit is ready to be executed
     */
    virtual bool IsReady() = 0;

    /**
     * Method that is executed on the engine
     * This method must be called after checking computation is ready
     */
    virtual void Compute() = 0;

    /**
     * Brings back reference of the atomic state counter for atomic comparison
     * of state counter
     * @return : reference of the state counter
     */
    std::atomic<std::size_t>& GetStateNum();

 protected:
    /**
     * Atomically increments state number
     */
    void incrementStateNum()
    {
        m_unitState.StateNum.fetch_add(1, std::memory_order_release);
    }

    /**
     * Atomically sets operation state to busy state (true)
     */
    void setBusy()
    {
        std::atomic_exchange_explicit(&m_unitState.IsBusy, true,
                                      std::memory_order_release);
    }

    /**
     * Atomically sets operation state to pending state (false)
     */
    void setReleased()
    {
        std::atomic_exchange_explicit(&m_unitState.IsBusy, false,
                                      std::memory_order_release);
    }

    /**
     * Checks if units surrounding the current unit is ready
     * @return
     */
    std::pair<bool, bool> checkForSurroundingUnits()
    {
        bool isPreviousReady = true, isNextReady = true;

        for (auto& previousUnitPtr : m_previousPtrVector)
            if (!previousUnitPtr || !previousUnitPtr->IsReady())
                isPreviousReady = false;

        if (!m_nextPtr || !m_nextPtr->IsReady())
            isNextReady = false;

        return std::pair(isPreviousReady, isNextReady);
    }

    /// UnitState object indicates execution state of ComputableUnit
    UnitState m_unitState;

    ComputableUnit* m_nextPtr;

    std::vector<ComputableUnit*> m_previousPtrVector;
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

    void SetPreviousPtr(ComputableUnit* computableUnitPtr)
    {
        if (ComputableUnit::m_previousPtrVector.empty())
            ComputableUnit::m_previousPtrVector.emplace_back(computableUnitPtr);
        else
            ComputableUnit::m_previousPtrVector.at(0) = computableUnitPtr;
    }

    void SetNextPtr(ComputableUnit* computableUnitPtr)
    {
        ComputableUnit::m_nextPtr = computableUnitPtr;
    }

    void Compute() override;

    bool IsReady() override;
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
     * Set or add next ComputableUnit ptr
     * @param computableUnitPtr : computablePtr to set
     */
    void SetNextPtr(CopyUnit* computableUnitPtr)
    {
        ComputableUnit::m_nextPtr = computableUnitPtr;
    }

    /**
     * Checks if source is ready
     * @return
     */
    bool IsReady() final;

 protected:
    TensorInfo m_outputTensorInfo;

    Tensor m_outputTensor;
};

/**
 * Unit that has no output, but has inputs
 * This type of unit plays role as sink of the computable graph
 */
class SinkUnit : public ComputableUnit
{
 public:
    /**
     * Constructor
     * @param inputTensorInfoVector : vector of tensorInfo to accept
     */
    explicit SinkUnit(std::vector<TensorInfo> inputTensorInfoVector);

    /**
     * Add previous computable Unit to this cell
     * @param computableUnitPtr
     */
    void AddPreviousPtr(CopyUnit* computableUnitPtr)
    {
        ComputableUnit::m_previousPtrVector.emplace_back(computableUnitPtr);
    }

    /**
     * Brings back if executableUnit is ready to be executed
     * @return : whether corresponding unit is ready to be executed
     */
    bool IsReady() final;

 protected:
    std::vector<TensorInfo> m_inputTensorInfoVector;
    std::vector<Tensor> m_inputTensorVector;
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

    void SetNextPtr(CopyUnit* computableUnitPtr)
    {
        ComputableUnit::m_nextPtr = computableUnitPtr;
    }

    void AddPreviousPtr(CopyUnit* computableUnitPtr)
    {
        ComputableUnit::m_previousPtrVector.emplace_back(computableUnitPtr);
    }

    bool IsReady() final;

 protected:
    std::vector<TensorInfo> m_inputTensorInfoVector;
    TensorInfo m_outputTensorInfo;

    Tensor m_outputTensor;
    std::vector<Tensor> m_inputTensorVector;
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTABLEUNIT_HPP
