// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/CopyUnit.hpp>

namespace CubbyDNN
{

CopyUnit::CopyUnit(CopyUnit&& copyUnit) noexcept
    : m_inputTensorIndex(copyUnit.m_inputTensorIndex),
      m_outputTensorIndex(copyUnit.m_outputTensorIndex),
      m_inputUnitPtr(std::move(copyUnit.m_inputUnitPtr)),
      m_outputUnitPtrVector(std::move(copyUnit.m_outputUnitPtrVector))
{
}

void CopyUnit::Forward()
{
    auto& inputTensor = m_inputUnitPtr->GetOutputForwardTensor();

    for (auto& outputTensor : m_outputForwardTensorVector)
        Tensor::CopyTensor(inputTensor, outputTensor);
}

void CopyUnit::Backward()
{
    const auto size = m_inputBackwardTensorVector.size();
    for (std::size_t idx = 0; idx < size; ++idx)
    {
        Tensor::CopyTensor(m_inputBackwardTensorVector.at(idx),
                           m_outputBackwardTensorVector.at(idx));
    }
}

bool CopyUnit::IsReady()
{
    const bool inputReady = m_inputUnitPtr->GetStateNum() ==
                            m_unitState.StateNum.load(std::memory_order_acquire)
                            + 1;

    bool outputReady = true;
    for (auto& outputPtr : m_outputUnitPtrVector)
    {
        outputReady = outputPtr->GetStateNum() ==
                      m_unitState.StateNum.load(std::memory_order_acquire);
    }

    return inputReady && outputReady;
}

void CopyUnit::ReleaseUnit()
{
    m_unitState.StateNum.fetch_add(1, std::memory_order_release);
}

std::size_t CopyUnit::GetStateNum() const
{
    return m_unitState.StateNum.load(std::memory_order_acquire);
}

} // namespace CubbyDNN
