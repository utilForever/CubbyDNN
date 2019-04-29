//
// Created by jwkim98 on 3/24/19.
//

#ifndef CUBBYDNN_LINKER_IMPL_HPP
#define CUBBYDNN_LINKER_IMPL_HPP

#include <cubbydnn/GraphUtil/Decl/Linker.hpp>
#include <cubbydnn/Tensors/Decl/TensorPlug.hpp>

namespace CubbyDNN
{
template <typename T>
Linker<T>::Linker(const TensorSocketPtr<T> socketPtr,
                  const TensorPlugPtr<T> plugPtr):
                  m_tensorPlugPtr(plugPtr), m_tensorSocketPtr(socketPtr),
                  m_PlugFuture(m_tensorPlugPtr.GetFuture()),
                  m_SocketFuture(m_tensorSocketPtr.GetFuture())
{
}

template <typename T>
bool Linker<T>::Link() const
{
    if (m_SocketFuture.valid() && m_PlugFuture.valid())
    {
        m_SocketFuture.wait();
        m_PlugFuture.wait();

        TensorDataPtr<T> oldPlugTensorPtr = m_tensorPlugPtr->GetDataPtr();
        TensorDataPtr<T> oldSocketTensorPtr = m_tensorSocketPtr->GetDataPtr();

        m_tensorPlugPtr->SetDataPtr(oldSocketTensorPtr);
        m_tensorSocketPtr->SetDataPtr(oldPlugTensorPtr);
    }
}
}  // namespace CubbyDNN

#endif  // CUBBYDNN_LINKER_IMPL_HPP
