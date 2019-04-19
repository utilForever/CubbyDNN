/**
 * @file : TensorSocket-impl.hpp
 * @author : Justin Kim
 */

#ifndef CUBBYDNN_TENSORSOCKET_IMPL_HPP
#define CUBBYDNN_TENSORSOCKET_IMPL_HPP

#include <cubbydnn/Tensors/Decl/TensorSocket.hpp>

namespace CubbyDNN
{
template <typename T>
TensorSocket<T>::TensorSocket()
    : m_futureReceive(m_promiseSend.get_future()),
      m_mtx(),
      m_lock(m_mtx),
      m_cond()
{
}

template <typename T>
void TensorSocket<T>::SendData(TensorDataPtr<T> tensorDataPtr)
{
    m_cond.wait(m_lock, [this](){return m_updateReady;});
    m_promiseSend.set_value(tensorDataPtr);
    m_updateReady = false;
}

template<typename T>
bool TensorSocket<T>::TrySendData(TensorDataPtr<T> tensorDataPtr)
{
    if(m_updateReady)
    {
        m_promiseSend.set_value(tensorDataPtr);
        m_updateReady = false;
        return true;
    }
    return false;

}

template <typename T>
TensorDataPtr<T> TensorSocket<T>::Request()
{
    if(m_futureReceive.valid())
    {
        m_futureReceive.wait();
        auto ptr = m_futureReceive.get();
        m_updateReady = true;
        m_cond.notify_all();
        return ptr;
    }
}

template <typename T>
TensorDataPtr<T> TensorSocket<T>::TryRequest()
{
    if (m_futureReceive.valid())
    {
        auto ptr = m_futureReceive.get();
        m_updateReady = true;
        m_cond.notify_all();
        return ptr;
    }

    return nullptr;
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSORSOCKET_IMPL_HPP
