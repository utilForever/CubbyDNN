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
{
}

template<typename T>
bool TensorSocket<T>::ReceiveData()
{
    if(m_tensorDataPtr)
    {
       m_promiseSend.set_value(m_tensorDataPtr);
       m_tensorDataPtr = nullptr;
       return true;
    }
    return false;
}

template<typename T>
TensorDataPtr<T> TensorSocket<T>::GetDataPtr() const noexcept
{
    return m_tensorDataPtr;
}

template<typename T>
bool TensorSocket<T>::SetDataPtr(TensorDataPtr<T> tensorDataPtr)
{
    if(!m_tensorDataPtr)
    {
        m_tensorDataPtr = tensorDataPtr;
        return true;
    }
    return false;
}

template<typename T>
std::future<TensorData<T>> TensorSocket<T>::GetFuture()
{
    return m_promiseSend.get_future();
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSORSOCKET_IMPL_HPP
