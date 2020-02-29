// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/SourceComputableUnits/SourceUnit.hpp>

namespace CubbyDNN
{
SourceUnit::SourceUnit(TensorInfo output, size_t numberOfOutputs)
    : ComputableUnit({}, output, UnitType::Source)
{
    m_outputPtrVector = std::vector<SharedPtr<ComputableUnit>>(
        numberOfOutputs, SharedPtr<ComputableUnit>());

    m_outputTensorVector.reserve(numberOfOutputs);
    for (size_t idx = 0; idx < numberOfOutputs; ++idx)
    {
        m_outputTensorVector.emplace_back(AllocateTensor(m_outputTensorInfo));
    }
}

bool SourceUnit::IsReady()
{
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

ConstantUnit::ConstantUnit(TensorInfo output, int numberOfOutputs,
                           void* dataPtr)
    : SourceUnit(output, numberOfOutputs),
      m_dataPtr(dataPtr)
{
    const auto byteSize = output.GetByteSize();
    assert(dataPtr != nullptr);
    m_byteSize = byteSize;
    for (auto& outputTensor : m_outputTensorVector)
    {
        std::memcpy(outputTensor.DataPtr, static_cast<void*>(m_dataPtr),
                    m_byteSize);
    }
}
} // namespace CubbyDNN
