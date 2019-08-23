//
// Created by jwkim98 on 8/13/19.
//

#ifndef CUBBYDNN_HIDDENUNIT_HPP
#define CUBBYDNN_HIDDENUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Units/CopyUnit.hpp>

namespace CubbyDNN
{
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
    size_t AddOutputPtr(CopyUnit* computableUnitPtr)
    {
        ComputableUnit::m_outputPtrVector.at(m_outputIndex) = computableUnitPtr;
        return m_outputIndex++;
    }

    //! Add previous computable Unit to this cell
    //! \param computableUnitPtr : computableUnitPtr to add
    void AddInputPtr(CopyUnit* computableUnitPtr, size_t index)
    {
        ComputableUnit::m_inputPtrVector.at(index) = computableUnitPtr;
    }

    //! Determines whether system is ready to compute
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

#endif  // CUBBYDNN_HIDDENUNIT_HPP
