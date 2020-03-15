// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTABLEUNIT_HPP
#define CUBBYDNN_COMPUTABLEUNIT_HPP

#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Tensors/TensorInfo.hpp>
#include <cubbydnn/Utils/SharedPtr.hpp>
#include <atomic>
#include <vector>
#include <string>

namespace CubbyDNN
{
class ComputableUnit
{
public:
    ComputableUnit(UnitType unitType);

    //! Constructor
    //! \param inputTensorInfoVector : vector of TensorInfo of inputs
    //! \param outputTensorInfo : vector of TensorInfo of outputs
    //! \param unitType : type of the unit
    ComputableUnit(std::vector<TensorInfo> inputTensorInfoVector,
                   TensorInfo outputTensorInfo, UnitType unitType);
    virtual ~ComputableUnit() = default;

    ComputableUnit(const ComputableUnit& computableUnit) = delete;
    ComputableUnit(ComputableUnit&& other) noexcept;

    ComputableUnit& operator=(ComputableUnit&& other) noexcept;
    ComputableUnit& operator=(const ComputableUnit& computableUnit) = delete;

    //! Adds output computable unit ptr to ComputableUnit
    //! \param computableUnitPtr : ptr to output computable unit
    std::size_t
    AddOutputPtr(const SharedPtr<ComputableUnit>& computableUnitPtr);

    //! Adds input computable unit ptr to this ComputableUnit
    //! \param computableUnitPtr : ptr to input computable unit
    //! \param index : indicates order of input argument.
    void AddInputPtr(const SharedPtr<ComputableUnit>& computableUnitPtr,
                     std::size_t index);

    //! Gets whether if executableUnit is ready to be executed
    //! \return : whether corresponding unit is ready to be executed
    virtual bool IsReady() = 0;

    //! Method that is executed on the engine
    //! This method must be called after checking computation is ready
    virtual void Compute() = 0;

    //! Called after computation for releasing the unit after computation
    //! Increments the stateNum and marks IsBusy as false
    void ReleaseUnit();

    //! Gets reference of the atomic state counter for atomic comparison
    //! of state counter
    //! \return : reference of the state counter
    std::size_t GetStateNum() const
    {
        return m_unitState.StateNum.load(std::memory_order_acquire);
    }

    virtual Tensor& GetInputTensor(std::size_t index)
    {
        return m_inputTensorVector.at(index);
    }

    virtual Tensor& GetOutputTensor(std::size_t index)
    {
        return m_outputTensorVector.at(index);
    }

    TensorInfo GetOutputTensorInfo() const
    {
        return m_outputTensorInfo;
    }


    UnitType Type = UnitType::Undefined;

protected:
    //! increments state number after execution
    void m_incrementStateNum()
    {
        m_unitState.StateNum.fetch_add(1, std::memory_order_release);
    }

    /// UnitState m_objectPtr indicates execution state of ComputableUnit
    UnitState m_unitState;
    /// ptr to units to receive result from
    std::vector<SharedPtr<ComputableUnit>> m_inputPtrVector;
    /// ptr to units to write result
    std::vector<SharedPtr<ComputableUnit>> m_outputPtrVector;
    /// vector to log states for debugging purpose
    std::vector<std::string> m_logVector;

    std::vector<TensorInfo> m_inputTensorInfoVector;
    TensorInfo m_outputTensorInfo;

    std::vector<Tensor> m_inputTensorVector;
    std::vector<Tensor> m_outputTensorVector;

private:
    std::size_t m_outputVectorIndex = 0;
};
}; // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTABLEUNIT_HPP
