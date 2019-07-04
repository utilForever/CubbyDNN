// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_DOUBLEBUFFER_HPP
#define CUBBYDNN_DOUBLEBUFFER_HPP

#include <cubbydnn/Utils/Sync.hpp>
#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Utils/SharedPtr-impl.hpp>

#include <condition_variable>
#include <future>
#include <memory>

/**
 *  This class performs role of socket that receives TensorObjects.
 *  This Class will be stored by Operations, and TensorObjects heading to that
 *  TensorSocket will point to corresponding TensorSocket by unique_ptr to
 * TensorSocket
 *
 */
namespace CubbyDNN
{
class DoubleBuffer
{
 public:
    explicit DoubleBuffer(const TensorInfo& tensorInfo, SyncPtr syncPtr);

    /**
     * Reassigning TensorSocket is disabled
     * @param tensorSocket
     * @return
     */
    DoubleBuffer& operator=(DoubleBuffer& tensorSocket) = delete;

    /**
     * Access method to front buffer
     * @return : reference of the front buffer
     */
    Tensor& GetFrontTensor();

    /**
     * Access method to back buffer
     * @return : reference of the back buffer
     */
    Tensor& GetBackTensor();

    /**
     * Swaps back and front tensors
     */
    void SwapTensors();


 private:

    Tensor m_frontTensor;

    Tensor m_backTensor;

    SyncPtr m_syncPtr;
};

class BufferPool
{
public:
    BufferPool(
            const std::vector<std::tuple<TensorInfo, SyncPtr>>& inputTensorInfoVector,
            std::tuple<TensorInfo, SyncPtr> outputTensorInfo);

    void SwapAll();

private:
    /// DoubleBuffer for receiving inputs
    std::vector<DoubleBuffer> m_inputBufferPool;
    /// DoubleBuffer for receiving outputs
    DoubleBuffer m_outputBuffer;
};

template <typename T>
using TensorSocketPtr = SharedPtr<DoubleBuffer>;

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TensorSocket_HPP
