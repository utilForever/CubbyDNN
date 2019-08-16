//
// Created by jwkim98 on 8/13/19.
//

#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>

namespace CubbyDNN
{
HiddenUnit::HiddenUnit(std::vector<TensorInfo> inputTensorInfoVector,
                       std::vector<TensorInfo> outputTensorInfoVector)
    : ComputableUnit(inputTensorInfoVector.size(),
                     outputTensorInfoVector.size()),
      m_inputTensorInfoVector(std::move(inputTensorInfoVector)),
      m_outputTensorInfoVector(std::move(outputTensorInfoVector))
{
    m_inputTensorVector.reserve(m_inputTensorInfoVector.size());
    for (auto& inputTensorInfo : m_inputTensorInfoVector)
    {
        m_inputTensorVector.emplace_back(AllocateTensor(inputTensorInfo));
    }

    m_outputTensorVector.reserve(m_outputTensorInfoVector.size());
    for (auto& outputTensorInfo : m_outputTensorInfoVector)
    {
        m_outputTensorVector.emplace_back(AllocateTensor(outputTensorInfo));
    }
}

HiddenUnit::HiddenUnit(HiddenUnit&& intermediateUnit) noexcept
    : ComputableUnit(std::move(intermediateUnit)),
      m_inputTensorInfoVector(
          std::move(intermediateUnit.m_inputTensorInfoVector)),
      m_outputTensorInfoVector(
          std::move(intermediateUnit.m_outputTensorInfoVector)),
      m_inputTensorVector(std::move(intermediateUnit.m_inputTensorVector)),
      m_outputTensorVector(std::move(intermediateUnit.m_outputTensorVector))
{
}

bool HiddenUnit::IsReady()
{
    if (ComputableUnit::m_unitState.IsBusy)
        return false;

    for (auto tensor : m_inputPtrVector)
    {
        if (tensor->GetStateNum() != this->GetStateNum() + 1)
            return false;
    }

    for (auto tensor : m_outputPtrVector)
    {
        if (tensor->GetStateNum() != this->GetStateNum())
            return false;
    }

    return true;
}

}  // namespace CubbyDNN
