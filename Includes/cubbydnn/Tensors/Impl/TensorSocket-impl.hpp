// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef CUBBYDNN_TENSORSOCKET_IMPL_HPP
#define CUBBYDNN_TENSORSOCKET_IMPL_HPP

#include <cubbydnn/Tensors/TensorSocket.hpp>

namespace CubbyDNN
{
template <typename T>
TensorSocket<T>::TensorSocket(SyncPtr operationSyncPtr, SyncPtr linkSyncPtr)
    : m_operationSyncPtr(operationSyncPtr), m_linkSyncPtr(linkSyncPtr)
{
}

template <typename T>
TensorPtr<T> TensorSocket<T>::MoveDataPtr() const noexcept
{
    auto tensorDataPtr = m_dataPtr;
    m_dataPtr = nullptr;
    return tensorDataPtr;
}

template <typename T>
bool TensorSocket<T>::SetDataPtrFromLinker(TensorPtr<T> tensorDataPtr)
{
    if (!m_dataPtr)
    {
        m_dataPtr = tensorDataPtr;
        /// Notify operation that data is ready to be executed
        m_operationSyncPtr->NotifyFinish();
        return true;
    }
    return false;
}

template<typename T>
bool TensorSocket<T>::SetDataPtrFromOperation(TensorPtr<T> tensorDataPtr)
{
    if (!m_dataPtr)
    {
        m_dataPtr = tensorDataPtr;
        /// Notify linker that data is ready to be swapped
        m_linkSyncPtr->NotifyFinish();
        return true;
    }
    return false;
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSORSOCKET_IMPL_HPP