/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_HIDDENUNIT_HPP
#define CUBBYDNN_HIDDENUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Computations/TensorOperations/TensorOperations.hpp>
#include <cubbydnn/Computations/TensorOperations/NaiveOperations.hpp>
#include <cubbydnn/Utils/SharedPtr.hpp>
#include <iostream>

namespace CubbyDNN
{
class HiddenUnit : public ComputableUnit
{
public:
    //! Constructor
    //! \param inputTensorInfoVector : vector of TensorInfo
    //! \param outputTensorInfo : TensorInfo of the output m_tensor
    //! \param numberOfOutputs : number of outputs that this unit is connected
    HiddenUnit(std::vector<TensorInfo> inputTensorInfoVector,
               TensorInfo outputTensorInfo, std::size_t numberOfOutputs = 1);
    ~HiddenUnit() = default;

    HiddenUnit(const HiddenUnit& hiddenUnit) = delete;
    HiddenUnit(HiddenUnit&& hiddenUnit) noexcept;

    HiddenUnit& operator=(const HiddenUnit& hiddenUnit) = delete;
    HiddenUnit& operator=(HiddenUnit&& hiddenUnit) noexcept;

    //! Adds output computable unit ptr to ComputableUnit
    //! \param computableUnitPtr : ptr to output computable unit
    std::size_t AddOutputPtr(
        const SharedPtr<ComputableUnit>& computableUnitPtr);

    //! Adds input computable unit ptr to this ComputableUnit
    //! \param computableUnitPtr : ptr to input computable unit
    //! \param index : indicates order of input argument.
    void AddInputPtr(const SharedPtr<ComputableUnit>& computableUnitPtr,
                     std::size_t index);

    virtual Tensor& GetInputForwardTensor(std::size_t index)
    {
        return m_inputForwardTensorVector.at(index);
    }

    virtual Tensor& GetOutputForwardTensor()
    {
        return m_outputForwardTensor;
    }

    virtual Tensor& GetInputBackwardTensor()
    {
        return m_inputBackwardTensor;
    }

    virtual Tensor& GetOutputBackwardTensor(std::size_t index)
    {
        return m_outputBackwardTensorVector.at(index);
    }

    TensorInfo GetOutputTensorInfo() const
    {
        return m_outputTensorInfo;
    }

    //! Determines whether system is ready to compute
    bool IsReady() final;

    //! Forward propagation
    virtual void Forward()
    {
    }

    //! Backward propagation
    virtual void Backward()
    {
    }

protected:
    std::unique_ptr<TensorOperation> m_tensorOperation = std::unique_ptr<
        NaiveOperation>();

private:
    
    /// ptr to units to receive result from
    std::vector<SharedPtr<ComputableUnit>> m_inputPtrVector;
    /// ptr to units to write result
    std::vector<SharedPtr<ComputableUnit>> m_outputPtrVector;
    /// vector to log states for debugging purpose
    std::vector<std::string> m_logVector;
    //! vector of tensor information for input in forward propagation
    std::vector<TensorInfo> m_inputTensorInfoVector;
    //! tensor information for output in forward propagation
    TensorInfo m_outputTensorInfo;
    //! vector of input tensors used to compute forward propagation
    std::vector<Tensor> m_inputForwardTensorVector;
    //! single output tensor of forward propagation
    Tensor m_outputForwardTensor;
    //! vector of output tensors used to compute back propagation
    Tensor m_inputBackwardTensor;
    //! single output tensor of back propagation
    std::vector<Tensor> m_outputBackwardTensorVector;
    std::size_t m_outputVectorIndex = 0;
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_HIDDENUNIT_HPP
