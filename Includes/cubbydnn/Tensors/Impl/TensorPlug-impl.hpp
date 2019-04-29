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
TensorPlug<T>::TensorPlug(const TensorShape& shape,
                          TensorSocketPtr<T> tensorSocketPtr)
    : m_info(TensorInfo(shape)), m_socket(tensorSocketPtr)
{
}

template <typename T>
TensorPlug<T>::TensorPlug(const TensorInfo& tensorInfo,
                          TensorSocketPtr<T> tensorSocketPtr)
    : m_info(tensorInfo), m_socket(tensorSocketPtr)
{
}

template <typename T>
TensorPlug<T>::TensorPlug(TensorPlug<T>&& obj) noexcept
{
    if (obj.m_data)
    {
        m_data = std::move(obj.m_data);
        m_info = obj.m_info;
    }
}

template <typename T>
TensorPlug<T>& TensorPlug<T>::operator=(TensorPlug<T>&& obj) noexcept
{
    if (*this == obj)
    {
        return *this;
    }

    if (obj.m_data)
    {
        m_data = std::move(obj.m_data);
        m_info = obj.m_info;
    }

    return *this;
}

template <typename T>
const TensorInfo& TensorPlug<T>::Info() const noexcept
{
    return m_info;
}

template <typename T>
bool TensorPlug<T>::SendData()
{
    if (m_data)
    {
        m_promise.set_value(m_data);
        m_data = nullptr;
        return true;
    }
    return false;
}

template <typename T>
TensorDataPtr<T> TensorPlug<T>::GetDataPtr() const noexcept
{
    return m_data;
}

template <typename T>
bool TensorPlug<T>::SetDataPtr(TensorDataPtr<T> tensorDataPtr)
{
    if (!m_data)
    {
        m_data = tensorDataPtr;
        return true;
    }
    return false;
}

template<typename T>
std::future<TensorData<T>> TensorPlug<T>::GetFuture()
{
    return m_promise.get_future();
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_OBJECT_IMPL_HPP