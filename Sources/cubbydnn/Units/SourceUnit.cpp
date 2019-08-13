//
// Created by jwkim98 on 8/13/19.
//

#include <cubbydnn/Units/SourceUnit.hpp>

namespace CubbyDNN
{
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
}  // namespace CubbyDNN
