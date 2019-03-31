// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_OBJECT_IMPL_HPP
#define CUBBYDNN_TENSOR_OBJECT_IMPL_HPP

#include <cubbydnn/GraphControl/Decl/Linker.hpp>
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
TensorObject<T>::TensorObject(const TensorObject<T>& obj)
{
    if (obj.m_data)
    {
        m_data = std::make_unique<TensorData<T>>(obj.Data(), obj.DataShape());
        m_info = obj.m_info;
    }
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
TensorObject<T>& TensorObject<T>::operator=(const TensorObject<T>& obj)
{
    if (*this == obj)
    {
        return *this;
    }

    if (obj.m_data)
    {
        m_data = std::make_unique<TensorData<T>>(obj.Data(), obj.DataShape());
        m_info = obj.m_info;
    }

    return *this;
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

template <typename T>
std::vector<T> TensorObject<T>::Data() const
{
    if (!m_data)
    {
        // TODO: Print error log
        return std::vector<T>();
    }

    return m_data->dataVec;
}

template <typename T>
std::unique_ptr<TensorData<T>> TensorObject<T>::DataPtr()
{
    if (!m_data || m_info.busy)
    {
        // TODO: Print error log
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(m_dataMtx);
    m_info.busy = true;
    return std::move(m_data);
}

template <typename T>
TensorShape TensorObject<T>::DataShape() const
{
    if (!m_data)
    {
        // TODO: Print error log
        return TensorShape();
    }

    return m_data->shape;
}

template <typename T>
void TensorObject<T>::MakeImmutable() const
{
    if (m_data && !m_info.busy)
    {
        m_data->isMutable = false;
    }
    else
    {
        // TODO: Print error log
    }
}
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_OBJECT_IMPL_HPP