//
// Created by jwkim98 on 6/28/19.
//

#include <cubbydnn/Utils/DoubleBuffer.hpp>

namespace CubbyDNN
{
DoubleBuffer::DoubleBuffer(const TensorInfo& tensorInfo, SyncPtr syncPtr)
    : m_frontTensor(AllocateTensor(tensorInfo)),
      m_backTensor(AllocateTensor(tensorInfo)),
      m_syncPtr(syncPtr)
{
}

Tensor& DoubleBuffer::GetFrontTensor()
{
    return m_frontTensor;
}

Tensor& DoubleBuffer::GetBackTensor()
{
    return m_backTensor;
}

void DoubleBuffer::SwapTensors()
{
    Tensor temp = std::move(m_frontTensor);
    m_frontTensor = std::move(m_backTensor);
    m_backTensor = std::move(temp);
}

BufferPool::BufferPool(
    const std::vector<std::tuple<TensorInfo, SyncPtr>>& inputTensorInfoVector,
    std::tuple<TensorInfo, SyncPtr> outputTensorInfo)
    : m_outputBuffer(DoubleBuffer(std::get<0>(outputTensorInfo),
                                  std::get<1>(outputTensorInfo)))
{
    m_inputBufferPool.reserve(inputTensorInfoVector.size());

    for (const auto& [inputTensorInfo, syncPtr] : inputTensorInfoVector)
    {
        m_inputBufferPool.emplace_back(DoubleBuffer(inputTensorInfo, syncPtr));
    }
}

void BufferPool::SwapAll()
{
    for (auto& inputDoubleBuffer : m_inputBufferPool)
    {
        inputDoubleBuffer.SwapTensors();
        // TODO : notify swap was done
    }

    m_outputBuffer.SwapTensors();
    // TODO : notify swap was done
}
}  // namespace CubbyDNN