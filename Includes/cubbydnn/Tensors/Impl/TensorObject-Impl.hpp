// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_OBJECT_IMPL_HPP
#define CUBBYDNN_TENSOR_OBJECT_IMPL_HPP

#include <cubbydnn/Tensors/Decl/TensorObject.hpp>

#include <cassert>

namespace CubbyDNN
{
template <typename T>
TensorObject<T>::TensorObject(std::size_t size, TensorShape shape, long from,
                              long to)
{
    std::vector<T> dataVector(size);
    assert(dataVector.size() == shape.Size());

    m_info = TensorInfo(from, to);
    m_data = std::make_unique<TensorData<T>>(dataVector, std::move(shape));
}



template <typename T>
TensorObject<T>::TensorObject(TensorObject<T>&& obj) noexcept
{
    if (obj.m_data)
    {
        m_data = std::move(obj.m_data);
        m_info = obj.m_info;
    }
}


template <typename T>
TensorObject<T>& TensorObject<T>::operator=(TensorObject<T>&& obj) noexcept
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
bool TensorObject<T>::operator==(const TensorObject<T>& obj) const
{
    return Info() == obj.Info();
}

template <typename T>
const TensorInfo& TensorObject<T>::Info() const
{
    return m_info;
}



template<typename T>
bool TensorObject<T>::SetData(TensorDataPtr<T> tensorDataPtr)
{
    if(!m_data)
    {
        if(!m_socket->TrySendData(tensorDataPtr))
            m_data = tensorDataPtr;
        return true;
    }

    if(!m_socket->SetData(m_data))
        return false;
    m_data = tensorDataPtr;
    return true;
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_OBJECT_IMPL_HPP