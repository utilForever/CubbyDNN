// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>
#include <cubbydnn/Utils/SharedPtr.hpp>

namespace CubbyDNN
{
HiddenUnit::HiddenUnit(std::vector<TensorInfo> inputTensorInfoVector,
                       TensorInfo outputTensorInfo)
    : ComputableUnit(UnitType::Hidden, std::move(inputTensorInfoVector), 
        std::move(outputTensorInfo))
{
    m_inputForwardTensorVector.reserve(m_inputTensorInfoVector.size());
    for (auto& inputTensorInfo : m_inputTensorInfoVector)
    {
        m_inputForwardTensorVector.
            emplace_back(AllocateTensor(inputTensorInfo));
    }

    m_outputForwardTensor = AllocateTensor(m_outputTensorInfo);
}

HiddenUnit::HiddenUnit(HiddenUnit&& hiddenUnit) noexcept
    : ComputableUnit(std::move(hiddenUnit))
{
    m_tensorOperation = std::move(hiddenUnit.m_tensorOperation);
}

HiddenUnit& HiddenUnit::operator=(HiddenUnit&& hiddenUnit) noexcept
{
    m_tensorOperation = std::move(hiddenUnit.m_tensorOperation);
    ComputableUnit::operator=(std::move(hiddenUnit));
    return *this;
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

std::size_t HiddenUnit::AddOutputPtr(
    const SharedPtr<ComputableUnit>& computableUnitPtr)
{
    m_outputPtrVector.emplace_back(computableUnitPtr);
    return m_outputPtrVector.size();
}

void HiddenUnit::AddInputPtr(
    const SharedPtr<ComputableUnit>& computableUnitPtr, std::size_t index)
{
    if (index >= m_inputPtrVector.size())
        throw std::runtime_error(
            "Number of inputs exceeds number given from declaration");

    m_inputPtrVector.at(index) = computableUnitPtr;
}
} // namespace CubbyDNN
