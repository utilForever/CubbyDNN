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
    ComputableUnit(size_t inputSize, size_t outputSize);

    ComputableUnit(ComputableUnit&& computableUnit);

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

    /// UnitState object indicates execution state of ComputableUnit
    UnitState m_unitState;

    std::vector<ComputableUnit*> m_inputPtrVector;

    std::vector<ComputableUnit*> m_outputPtrVector;
};

/**
 * CopyUnits
 * This will connect between other units(not copy) by copying the output of
 * previous unit to input of next unit
 */
class CopyUnit : public ComputableUnit
{
 public:
    CopyUnit();

    CopyUnit(CopyUnit&& copyUnit) noexcept;

    /**
     * Sets ComputableUnitPtr of previous unit to copy from
     * If no computableUnitPtr has been assigned, unit is added. If it has
     * previously assigned computableUnitPtr, computableUnitPtr is replaced by
     * given parameter
     * @param computableUnitPtr : computableUnitPtr to add or replace
     */
    void SetOriginPtr(ComputableUnit* computableUnitPtr)
    {
        if (ComputableUnit::m_inputPtrVector.empty())
            ComputableUnit::m_inputPtrVector.emplace_back(computableUnitPtr);
        else
            ComputableUnit::m_inputPtrVector.at(0) = computableUnitPtr;
    }

    /**
     * Sets ComputableUnitPtr of previous unit to copy from
     * If no computableUnitPtr has been assigned, unit is added. If it has
     * previously assigned computableUnitPtr, computableUnitPtr is replaced by
     * given parameter
     * @param computableUnitPtr : computableUnitPtr to add or replace
     */
    void SetDestinationPtr(ComputableUnit* computableUnitPtr)
    {
        if (ComputableUnit::m_outputPtrVector.empty())
            ComputableUnit::m_outputPtrVector.emplace_back(computableUnitPtr);
        else
            ComputableUnit::m_outputPtrVector.at(0) = computableUnitPtr;
    }

    /**
     * Computation
     */
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
     * @param outputTensorInfoVector : TensorInfo of the output tensor(Which is
     * always less than 1)
     */
    explicit SourceUnit(std::vector<TensorInfo> outputTensorInfoVector);

    SourceUnit(SourceUnit&& sourceUnit) noexcept;

    /**
     * Set or add next ComputableUnit ptr
     * @param computableUnitPtr : computablePtr to set
     */
    void AddOutputPtr(CopyUnit* computableUnitPtr)
    {
        ComputableUnit::m_outputPtrVector.emplace_back(computableUnitPtr);
    }

    /**
     * Checks if source is ready
     * @return : true if ready to be computed false otherwise
     */
    bool IsReady() final;

    void Compute() override{}

 protected:
    std::vector<TensorInfo> m_outputTensorInfoVector;
    std::vector<Tensor> m_outputTensorVector;
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
    explicit SinkUnit(std::vector<TensorInfo> inputTensorInfoVector,
                      size_t inputSize);

    SinkUnit(SinkUnit&& sinkUnit) noexcept;

    /**
     * Add previous computable Unit to this cell
     * @param computableUnitPtr : computableUnitPtr to add
     */
    void AddInputPtr(CopyUnit* computableUnitPtr, size_t index)
    {
        ComputableUnit::m_inputPtrVector.at(index) = computableUnitPtr;
    }

    /**
     * Brings back if executableUnit is ready to be executed
     * @return : whether corresponding unit is ready to be executed
     */
    bool IsReady() final;

    void Compute() override{}

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
     * @param outputTensorInfoVector : TensorInfo of the output tensor
     */
    IntermediateUnit(std::vector<TensorInfo> inputTensorInfoVector,
                     std::vector<TensorInfo> outputTensorInfoVector);

    IntermediateUnit(IntermediateUnit&& intermediateUnit) noexcept;

    /**
     * Add next computable Unit to this cell
     * @param computableUnitPtr : computableUnitPtr to add
     */
    void AddOutputPtr(CopyUnit* computableUnitPtr)
    {
        ComputableUnit::m_outputPtrVector.emplace_back(computableUnitPtr);
    }

    /**
     * Add previous computable Unit to this cell
     * @param computableUnitPtr : computableUnitPtr to add
     */
    void AddInputPtr(CopyUnit* computableUnitPtr, size_t index)
    {
        ComputableUnit::m_outputPtrVector.at(index) = computableUnitPtr;
    }

    /**
     * @return
     */
    bool IsReady() final;

    void Compute() override {}

 protected:
    std::vector<TensorInfo> m_inputTensorInfoVector;
    std::vector<TensorInfo> m_outputTensorInfoVector;

    std::vector<Tensor> m_inputTensorVector;
    std::vector<Tensor> m_outputTensorVector;
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTABLEUNIT_HPP
