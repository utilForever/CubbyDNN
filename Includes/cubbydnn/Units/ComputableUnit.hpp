// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTABLEUNIT_HPP
#define CUBBYDNN_COMPUTABLEUNIT_HPP

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Tensors/TensorInfo.hpp>
#include <cubbydnn/Utils/SharedPtr-impl.hpp>

namespace CubbyDNN
{
//! UnitState
//! Wrapper class containing the state and StateNum
//! This represents the execution state of computable Unit
struct UnitState
{
    explicit UnitState();
    /// State number of current
    std::atomic<std::size_t> StateNum = 0;
    /// True if unit is already in the task queue
    std::atomic<bool> IsBusy = false;
};

class ComputableUnit
{
 public:
    //! Default constructor
    //! Initializes unitInfo with pending state
    //! \param inputSize : size of the input for this unit
    //! \param outputSize : size of the output for this unit
    ComputableUnit(size_t inputSize, size_t outputSize);

    //! Move constructor
    //! \param computableUnit : ComputableUnit to move from
    ComputableUnit(ComputableUnit&& computableUnit) noexcept;

    virtual ~ComputableUnit() = default;

    //! Called before computation for acquiring the unit in order to compute
    //! Marks IsBusy as True in order to prevent same tasks being enqueued
    //! multiple times
    void AcquireUnit()
    {
        setBusy();
    }

    //! Called after computation for releasing the unit after computation
    //! Increments the stateNum and marks IsBusy as false
    void ReleaseUnit()
    {
        incrementStateNum();
        setReleased();
    }

    //! Brings back if executableUnit is ready to be executed
    //! \return : whether corresponding unit is ready to be executed
    virtual bool IsReady() = 0;

    //! Method that is executed on the engine
    //! This method must be called after checking computation is ready
    virtual void Compute() = 0;

    //! Brings back reference of the atomic state counter for atomic comparison
    //! of state counter
    //! \return : reference of the state counter
    std::atomic<std::size_t>& GetStateNum();

 protected:
    //! increments state number after execution
    void incrementStateNum()
    {
        m_unitState.StateNum.fetch_add(1, std::memory_order_relaxed);
        std::cout << "Increment" << std::endl;
    }

    //! Atomically sets isBusy to true
    void setBusy()
    {
        std::atomic_exchange_explicit(&m_unitState.IsBusy, true,
                                      std::memory_order_relaxed);
    }

    //! Atomically sets operation state to pending state (false)
    void setReleased()
    {
        std::atomic_exchange_explicit(&m_unitState.IsBusy, false,
                                      std::memory_order_relaxed);
    }

    virtual Tensor& GetInputTensor(size_t index) = 0;

    virtual Tensor& GetOutputTensor(size_t index) = 0;

    /// UnitState object indicates execution state of ComputableUnit
    UnitState m_unitState;
    /// ptr to units to receive result from
    std::vector<ComputableUnit*> m_inputPtrVector;
    /// ptr to units to write result
    std::vector<ComputableUnit*> m_outputPtrVector;
    /// vector to log states for debugging purpose
    std::vector<std::string> m_logVector;

    size_t m_inputIndex = 0;
    size_t m_outputIndex = 0;

    Tensor tensor = Tensor(nullptr, TensorInfo({ 0 }));
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

    //! Sets ComputableUnitPtr of previous unit to copy from
    //! If no computableUnitPtr has been assigned, unit is added. If it has
    //! previously assigned computableUnitPtr, computableUnitPtr is replaced by
    //! given parameter
    //! \param computableUnitPtr : computableUnitPtr to add or replace
    void SetInputPtr(ComputableUnit* computableUnitPtr)
    {
        ComputableUnit::m_inputPtrVector.at(m_inputIndex++) = computableUnitPtr;
    }

    //! Sets ComputableUnitPtr of previous unit to copy from
    //! If no computableUnitPtr has been assigned, unit is added. If it has
    //! previously assigned computableUnitPtr, computableUnitPtr is replaced by
    //! given parameter
    //! \param computableUnitPtr : computableUnitPtr to add or replace
    void SetOutputPtr(ComputableUnit* computableUnitPtr)
    {
        ComputableUnit::m_outputPtrVector.at(m_outputIndex++) =
            computableUnitPtr;
    }

    Tensor& GetInputTensor(size_t index) override
    {
        assert(false);
        return tensor;
    }

    Tensor& GetOutputTensor(size_t index) override
    {
        assert(false);
        return tensor;
    }

    //! Implements copy operation between input and output
    void Compute() override;
    //! Checks if this copyUnit is ready to be executed
    bool IsReady() override;
};

//! Unit that has no input, but has output only.
//! This type of unit must be able to fetch data(from the disk or cache)
//! or generator
class SourceUnit : public ComputableUnit
{
 public:
    //! Constructor
    //! \param outputTensorInfoVector : TensorInfo of the output tensor(Which is
    //! always less than 1)
    explicit SourceUnit(std::vector<TensorInfo> outputTensorInfoVector);

    SourceUnit(SourceUnit&& sourceUnit) noexcept;

    //! Set or add next ComputableUnit ptr
    //! \param computableUnitPtr : computablePtr to set
    void AddOutputPtr(CopyUnit* computableUnitPtr)
    {
        ComputableUnit::m_outputPtrVector.at(m_outputIndex++) =
            computableUnitPtr;
    }

    //! Checks if source is ready
    //! \return : true if ready to be computed false otherwise
    bool IsReady() final;

    void Compute() override
    {
        std::cout << "SourceUnit" << std::endl;
        std::cout << m_unitState.StateNum << std::endl;
    }

    Tensor& GetInputTensor(size_t index) override
    {
        return tensor;
    }

    Tensor& GetOutputTensor(size_t index) override
    {
        return m_outputTensorVector.at(index);
    }

 private:
    std::vector<TensorInfo> m_outputTensorInfoVector;
    std::vector<Tensor> m_outputTensorVector;
};

//! Unit that has no output, but has inputs
//! This type of unit plays role as sink of the computable graph
class SinkUnit : public ComputableUnit
{
 public:
    //! Constructor
    //! \param inputTensorInfoVector : vector of tensorInfo to accept
    explicit SinkUnit(std::vector<TensorInfo> inputTensorInfoVector);

    SinkUnit(SinkUnit&& sinkUnit) noexcept;

    //! Add previous computable Unit to this cell
    //! \param computableUnitPtr : computableUnitPtr to add
    void AddInputPtr(CopyUnit* computableUnitPtr, size_t index)
    {
        ComputableUnit::m_inputPtrVector.at(index) = computableUnitPtr;
    }

    //! Brings back if executableUnit is ready to be executed
    //! \return : whether corresponding unit is ready to be executed
    bool IsReady() final;

    void Compute() override
    {
        std::cout << "SinkUnit" << std::endl;
        std::cout << m_unitState.StateNum << std::endl;
    }

    Tensor& GetInputTensor(size_t index) override
    {
        return m_inputTensorVector.at(index);
    }

    Tensor& GetOutputTensor(size_t index) override
    {
        return tensor;
    }

 protected:
    std::vector<TensorInfo> m_inputTensorInfoVector;
    std::vector<Tensor> m_inputTensorVector;
};

class HiddenUnit : public ComputableUnit
{
 public:
    //! Constructor
    //! \param inputTensorInfoVector : vector of TensorInfo
    //! \param outputTensorInfoVector : TensorInfo of the output tensor
    HiddenUnit(std::vector<TensorInfo> inputTensorInfoVector,
               std::vector<TensorInfo> outputTensorInfoVector);

    HiddenUnit(HiddenUnit&& intermediateUnit) noexcept;

    //! Add next computable Unit to this cell
    //! \param computableUnitPtr : computableUnitPtr to add
    void AddOutputPtr(CopyUnit* computableUnitPtr)
    {
        ComputableUnit::m_outputPtrVector.at(m_outputIndex) = computableUnitPtr;
    }

    //! Add previous computable Unit to this cell
    //! \param computableUnitPtr : computableUnitPtr to add
    void AddInputPtr(CopyUnit* computableUnitPtr, size_t index)
    {
        ComputableUnit::m_inputPtrVector.at(index) = computableUnitPtr;
    }

    /**
     * @return
     */
    bool IsReady() final;

    void Compute() override
    {
        std::cout << "HiddenUnit" << std::endl;
        std::cout << m_unitState.StateNum << std::endl;
    }

    Tensor& GetInputTensor(size_t index) override
    {
        return m_inputTensorVector.at(index);
    }

    Tensor& GetOutputTensor(size_t index) override
    {
        return m_outputTensorVector.at(index);
    }

 protected:
    std::vector<TensorInfo> m_inputTensorInfoVector;
    std::vector<TensorInfo> m_outputTensorInfoVector;

    std::vector<Tensor> m_inputTensorVector;
    std::vector<Tensor> m_outputTensorVector;
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTABLEUNIT_HPP
