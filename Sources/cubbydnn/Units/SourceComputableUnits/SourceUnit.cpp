// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/SourceComputableUnits/SourceUnit.hpp>

namespace CubbyDNN
{
SourceUnit::SourceUnit(std::vector<TensorInfo> outputTensorInfoVector)
    : ComputableUnit({}, std::move(outputTensorInfoVector), UnitType::Source)
{
    m_outputPtrVector = std::vector<SharedPtr<ComputableUnit>>(
        m_outputTensorInfoVector.size(), SharedPtr<ComputableUnit>());

    m_outputTensorVector.reserve(outputTensorInfoVector.size());
    for (const auto& tensorInfo : m_outputTensorInfoVector)
    {
        m_outputTensorVector.emplace_back(AllocateTensor(tensorInfo));
    }
}

bool SourceUnit::IsReady()
{
    if (m_unitState.IsBusy)
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

ConstantUnit::ConstantUnit(TensorInfo output, int numberOfOutputs, void* dataPtr)
    : SourceUnit(std::vector<TensorInfo>(numberOfOutputs, output)),
      m_dataPtr(dataPtr)
{
    const auto byteSize = output.ByteSize();
    assert(dataPtr != nullptr);
    m_byteSize = byteSize;
    for (auto& outputTensor : m_outputTensorVector)
    {
        std::memcpy(outputTensor.DataPtr, static_cast<void*>(m_dataPtr),
                    m_byteSize);
    }
}
} // namespace CubbyDNN
