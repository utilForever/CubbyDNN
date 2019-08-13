// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
UnitState::UnitState() : StateNum(0), IsBusy(false)
{
}

ComputableUnit::ComputableUnit(size_t inputSize, size_t outputSize)
    : m_inputPtrVector(inputSize), m_outputPtrVector(outputSize)
{
    m_inputPtrVector.reserve(inputSize);
    m_outputPtrVector.reserve(outputSize);
}

ComputableUnit::ComputableUnit(ComputableUnit&& computableUnit) noexcept
    : m_inputPtrVector(std::move(computableUnit.m_inputPtrVector)),
      m_outputPtrVector(std::move(computableUnit.m_outputPtrVector)),
      m_logVector(std::move(computableUnit.m_logVector))
{
}

std::atomic<std::size_t>& ComputableUnit::GetStateNum()
{
    return m_unitState.StateNum;
}

CopyUnit::CopyUnit() : ComputableUnit(1, 1)
{
}

CopyUnit::CopyUnit(CopyUnit&& copyUnit) noexcept
    : ComputableUnit(std::move(copyUnit))
{
}

void CopyUnit::Compute()
{
    std::cout << "CopyUnit" << std::endl;
    std::cout << m_unitState.StateNum << std::endl;

    auto& inputTensor =
        m_inputPtrVector.at(0)->GetOutputTensor(m_inputTensorIndex);

    auto& outputTensor =
        m_outputPtrVector.at(0)->GetInputTensor(m_outputTensorIndex);
    CopyTensor(inputTensor, outputTensor);
}

bool CopyUnit::IsReady()
{
    if (ComputableUnit::m_unitState.IsBusy)
        return false;

    auto& stateNum = GetStateNum();
    return (ComputableUnit::m_inputPtrVector.at(0)->GetStateNum() ==
                (stateNum + 1) &&
            ComputableUnit::m_outputPtrVector.at(0)->GetStateNum() == stateNum);
}

SourceUnit::SourceUnit(std::vector<TensorInfo> outputTensorInfoVector)
    : ComputableUnit(1, outputTensorInfoVector.size()),
      m_outputTensorInfoVector(std::move(outputTensorInfoVector))
{
    m_outputTensorVector.reserve(outputTensorInfoVector.size());
    for (const auto& tensorInfo : m_outputTensorInfoVector)
    {
        m_outputTensorVector.emplace_back(AllocateTensor(tensorInfo));
    }
}

SourceUnit::SourceUnit(SourceUnit&& sourceUnit) noexcept
    : ComputableUnit(std::move(sourceUnit)),
      m_outputTensorInfoVector(std::move(sourceUnit.m_outputTensorInfoVector)),
      m_outputTensorVector(std::move(sourceUnit.m_outputTensorVector))
{
}

bool SourceUnit::IsReady()
{
    if (ComputableUnit::m_unitState.IsBusy)
        return false;

    auto isReady = true;
    for (const auto& nextPtr : m_outputPtrVector)
    {
        if (nextPtr->GetStateNum() != GetStateNum())
        {
            isReady = false;
            break;
        }
    }
    return isReady;
}

SinkUnit::SinkUnit(std::vector<TensorInfo> inputTensorInfoVector)
    : ComputableUnit(inputTensorInfoVector.size(), 1),
      m_inputTensorInfoVector(std::move(inputTensorInfoVector))
{
    m_inputTensorVector.reserve(m_inputTensorInfoVector.size());
    for (const auto& tensorInfo : m_inputTensorInfoVector)
    {
        m_inputTensorVector.emplace_back(AllocateTensor(tensorInfo));
    }
}

SinkUnit::SinkUnit(SinkUnit&& sinkUnit) noexcept
    : ComputableUnit(std::move(sinkUnit)),
      m_inputTensorInfoVector(std::move(sinkUnit.m_inputTensorInfoVector)),
      m_inputTensorVector(std::move(sinkUnit.m_inputTensorVector))
{
}

bool SinkUnit::IsReady()
{
    if (ComputableUnit::m_unitState.IsBusy)
        return false;
    for (const auto& previousPtr : m_inputPtrVector)
    {
        if (previousPtr->GetStateNum() != GetStateNum() + 1)
            return false;
    }
    return true;
}

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