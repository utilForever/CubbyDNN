// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_OBJECT_IMPL_HPP
#define CUBBYDNN_TENSOR_OBJECT_IMPL_HPP

#include <cubbydnn/Tensors/Decl/TensorPlug.hpp>

#include <cassert>

namespace CubbyDNN
{

template <typename T>
TensorPlug<T>::TensorPlug(SyncPtr operationSyncPtr, SyncPtr linkSyncPtr)
    : m_operationSyncPtr(operationSyncPtr), m_linkSyncPtr(linkSyncPtr)
{
}

template <typename T>
TensorDataPtr<T> TensorPlug<T>::MoveDataPtr() const noexcept
{
    auto tensorDataPtr = m_dataPtr;
    m_dataPtr = nullptr;
    return tensorDataPtr;
}

template <typename T>
bool TensorPlug<T>::SetDataPtrFromLinker(TensorDataPtr<T> tensorDataPtr)
{
    if (!m_dataPtr)
    {
        m_dataPtr = tensorDataPtr;
        m_operationSyncPtr->NotifyFinish();
        return true;
    }
    return false;
}

template <typename T>
bool TensorPlug<T>::SetDataPtrFromOperation(TensorDataPtr<T> tensorDataPtr)
{
    if (!m_dataPtr)
    {
        m_dataPtr = tensorDataPtr;
        m_linkSyncPtr->NotifyFinish();
        return true;
    }
    return false;
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_OBJECT_IMPL_HPP