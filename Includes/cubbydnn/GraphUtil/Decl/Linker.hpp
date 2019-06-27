/**
 * Copyright (c) 2019 Chris Ohk, Justin Kim
 * @file : Linker.hpp
 * @brief : helper functions that link TensorObjects and Operations
 */

#ifndef CUBBYDNN_LINKER_HPP
#define CUBBYDNN_LINKER_HPP

#include <cubbydnn/GraphUtil/Decl/Sync.hpp>
#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Tensors/TensorPlug.hpp>
#include <cubbydnn/Tensors/TensorSocket.hpp>

#include <memory>

namespace CubbyDNN
{
/**
 * This class Connects a single TensorPlug and TensorSocket
 * This class synchronizes operations and swaps data between them
 * @tparam T
 */
template <typename T>
class Linker : virtual IExecutable
{
 public:
    /**
     * Constructor
     * @param socketPtr : pointer of TensorSocket to link
     * @param plugPtr : pointer of TensorPlug to link
     * @param syncPtr : pointer of Sync object to use
     */
    Linker(TensorSocketPtr<T> socketPtr, TensorPlugPtr<T> plugPtr,
           SyncPtr syncPtr);

    /**
     * Swap
     * Swaps DataPtr of TensorPlugPtr and TensorSocketPtr
     */
    void Swap();

    /**
     * Invoke
     * Starts synchronization thread that waits for tensorPlugs and tensorSockets to finish
     */
    void Start() final;

    /**
     * Finish
     * Finishes synchronization thread that waits for tensorPlugs and tensorSockets to finish
     */
    void Finish() final;

 private:
    /// Ptr to tensorPlug
    TensorPlugPtr<T> m_tensorPlugPtr;
    /// Ptr to tensorSocket
    TensorSocketPtr<T> m_tensorSocketPtr;
    /// Used to manage atomic-counter based synchronization
    SyncPtr m_syncPtr;
    std::thread m_thread;
    bool m_forceFinish = false;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_LINKER_HPP
