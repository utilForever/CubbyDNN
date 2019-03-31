//
// Created by jwkim98 on 3/31/19.
//

#ifndef CUBBYDNN_TENSORSOCKET_IMPL_HPP
#define CUBBYDNN_TENSORSOCKET_IMPL_HPP

#include <cubbydnn/Tensors/Decl/TensorSocket.hpp>

namespace CubbyDNN
{
template <typename T>
void TensorSocket<T>::ReceiveData(TensorDataPtr<T> tensorDataPtr)
{
    m_promiseSend.set_value(tensorDataPtr);
}

template <typename T>
TensorSocket<T>::TensorSocket()
{
    m_futureReceive = m_promiseSend.get_future();
}

template <typename T>
TensorDataPtr<T> TensorSocket<T>::Request()
{
    return m_futureReceive.get();
}

template <typename T>
TensorDataPtr<T> TensorSocket<T>::TryRequest()
{
    if(m_futureReceive.valid())
    {
        return m_futureReceive.get();
    }

    return nullptr;
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSORSOCKET_IMPL_HPP
