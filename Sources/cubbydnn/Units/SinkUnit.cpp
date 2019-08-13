//
// Created by jwkim98 on 8/13/19.
//

#include <cubbydnn/Units/SinkUnit.hpp>

namespace CubbyDNN
{
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
}  // namespace CubbyDNN