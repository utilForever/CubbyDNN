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

    for (const auto& outputPtrIndexPair : m_outputUnitPtrVector)
    {
        const auto& outputUnitPtr = outputPtrIndexPair.ptr;
        const auto outputIndex = outputPtrIndexPair.index;
        Tensor::CopyTensor(inputTensor,
                           outputUnitPtr->GetInputForwardTensor(outputIndex));
    }
}

void CopyUnit::Backward()
{
    auto& inputTensor = m_inputUnitPtr->GetOutputForwardTensor();

    for (const auto& outputPtrIndexPair : m_outputUnitPtrVector)
    {
        const auto& outputUnitPtr = outputPtrIndexPair.ptr;
        const auto outputIndex = outputPtrIndexPair.index;
        // TODO : Add output tensors and copy them to source
        outputUnitPtr->GetInputForwardTensor(outputIndex);
    }
}

std::size_t CopyUnit::GetStateNum() const
{
    return m_unitState.StateNum.load(std::memory_order_acquire);
}
} // namespace CubbyDNN
