//
// Created by jwkim98 on 8/13/19.
//

#ifndef CUBBYDNN_SINKUNIT_HPP
#define CUBBYDNN_SINKUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Units/CopyUnit.hpp>

namespace CubbyDNN
{
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
    void AddInputPtr(ComputableUnit* computableUnitPtr, size_t index)
    {
        m_inputPtrVector.at(index) = SharedPtr<ComputableUnit>::Make(computableUnitPtr);
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
        index;
        return tensor;
    }

 protected:
    std::vector<TensorInfo> m_inputTensorInfoVector;
    std::vector<Tensor> m_inputTensorVector;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_SINKUNIT_HPP
