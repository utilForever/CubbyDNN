// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTABLEUNIT_HPP
#define CUBBYDNN_COMPUTABLEUNIT_HPP

#include <atomic>
#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Tensors/TensorInfo.hpp>

namespace CubbyDNN
{
class ComputableUnit
{
public:
    //! \param unitType : type of the unit
    //! \param inputTensorInfoVector : vector of input tensor information
    //! \param outputTensorInfo : output tensor information
    ComputableUnit(UnitType unitType,
                   std::vector<TensorInfo> inputTensorInfoVector,
                   TensorInfo outputTensorInfo);

    //! Constructor
    virtual ~ComputableUnit() = default;

    ComputableUnit(const ComputableUnit& computableUnit) = delete;
    ComputableUnit(ComputableUnit&& other) noexcept;

    ComputableUnit& operator=(ComputableUnit&& other) noexcept;
    ComputableUnit& operator=(const ComputableUnit& computableUnit) = delete;

    UnitIdentifier GetIdentifier()
    {
        return m_identifier;
    }

    virtual Tensor& GetInputForwardTensor(std::size_t index)
    {
        return m_inputForwardTensorVector.at(index);
    }

    virtual Tensor& GetOutputForwardTensor()
    {
        return m_outputForwardTensor;
    }

    virtual Tensor& GetInputBackwardTensor(std::size_t index)
    {
        return m_outputBackwardTensorVector.at(index);
    }

    virtual Tensor& GetOutputBackwardTensor(std::size_t index)
    {
        return m_inputBackwardTensorVector.at(index);
    }

    TensorInfo GetOutputTensorInfo() const
    {
        return m_outputTensorInfo;
    }

    const std::vector<UnitIdentifier>& GetInputUnitVector() const
    {
        return m_inputUnitVector;
    }

    const std::vector<UnitIdentifier>& GetOutputUnitVector() const
    {
        return m_outputUnitVector;
    }

    void SetInputUnitVector(std::vector<UnitIdentifier> inputUnitVector)
    {
        m_inputUnitVector = inputUnitVector;
    }

    void AddOutputUnitVector(UnitIdentifier outputUnit)
    {
        m_outputUnitVector.emplace_back(outputUnit);
    }

    //! Gets whether if executableUnit is ready to be executed
    //! \return : whether corresponding unit is ready to be executed
    virtual bool IsReady() = 0;

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

    UnitType Type = UnitType::Undefined;

protected:
    /// UnitState m_objectPtr indicates execution state of ComputableUnit
    UnitState m_unitState;
   UnitIdentifier m_identifier;
    std::vector<UnitIdentifier> m_inputUnitVector;
    std::vector<UnitIdentifier> m_outputUnitVector;
    //! vector of tensor information for input in forward propagation
    std::vector<TensorInfo> m_inputTensorInfoVector;
    //! tensor information for output in forward propagation
    TensorInfo m_outputTensorInfo;
    //! vector of input tensors used to compute forward propagation
    std::vector<Tensor> m_inputForwardTensorVector;
    //! single output tensor of forward propagation
    Tensor m_outputForwardTensor;
    //! single output tensor of back propagation
    std::vector<Tensor> m_inputBackwardTensorVector;
    //! vector of output tensors used to compute back propagation
    std::vector<Tensor> m_outputBackwardTensorVector;
};
}; // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTABLEUNIT_HPP
