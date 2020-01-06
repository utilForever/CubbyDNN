// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>

namespace CubbyDNN
{
HiddenUnit::HiddenUnit(std::vector<TensorInfo> inputTensorInfoVector,
                       std::vector<TensorInfo> outputTensorInfoVector)
    : ComputableUnit(inputTensorInfoVector, outputTensorInfoVector,
                     UnitType::Hidden)
{
    m_outputPtrVector =
        std::vector<SharedPtr<ComputableUnit>>(m_outputTensorInfoVector.size());
    m_inputPtrVector =
        std::vector<SharedPtr<ComputableUnit>>(m_inputTensorInfoVector.size());

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

bool HiddenUnit::IsReady()
{
    if (m_unitState.IsBusy)
        return false;

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

}  // namespace CubbyDNN
