// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Computations/Functions/Matrix.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>

namespace CubbyDNN
{
HiddenUnit::HiddenUnit(std::vector<TensorInfo> inputTensorInfoVector,
                      TensorInfo outputTensorInfo, std::size_t numberOfOutputs)
    : ComputableUnit(std::move(inputTensorInfoVector),
                     outputTensorInfo, UnitType::Hidden)
{
    m_outputPtrVector =
        std::vector<SharedPtr<ComputableUnit>>(numberOfOutputs);
    m_inputPtrVector =
        std::vector<SharedPtr<ComputableUnit>>(m_inputTensorInfoVector.size());

    m_inputTensorVector.reserve(m_inputTensorInfoVector.size());
    for (auto& inputTensorInfo : m_inputTensorInfoVector)
    {
        m_inputTensorVector.emplace_back(AllocateTensor(inputTensorInfo));
    }

    m_outputTensorVector.reserve(numberOfOutputs);
    for (std::size_t idx = 0; idx < numberOfOutputs; ++ idx)
    {
        m_outputTensorVector.emplace_back(AllocateTensor(m_outputTensorInfo));
    }
}

bool HiddenUnit::IsReady()
{
    for (auto& elem : m_inputPtrVector)
    {
        if (elem->GetStateNum() != this->GetStateNum() + 1)
            return false;
    }

    for (auto& elem : m_outputPtrVector)
    {
        if (elem->GetStateNum() != this->GetStateNum())
            return false;
    }

    return true;
}

MatMul::MatMul(const TensorInfo& inputA, const TensorInfo& inputB,
               const TensorInfo& output, std::size_t numberOfOutputs)
    : HiddenUnit({ inputA, inputB },  output, numberOfOutputs)
{
    assert(inputA.GetShape().Col == inputB.GetShape().Row);
    assert(inputA.GetShape().Batch == inputB.GetShape().Batch &&
        inputA.GetShape().Batch == output.GetShape().Batch);;
    assert(inputA.GetShape().Channel == inputB.GetShape().Channel &&
        inputA.GetShape().Channel == output.GetShape().Channel);
}

void MatMul::Compute()
{
    MultiplyOp(m_inputTensorVector.at(0), m_inputTensorVector.at(1),
             m_outputTensorVector.at(0));
}
} // namespace CubbyDNN
