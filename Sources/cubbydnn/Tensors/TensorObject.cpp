// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Tensors/TensorObject.hpp>

#include <cassert>

namespace CubbyDNN
{
TensorObject::TensorObject(std::size_t size, TensorShape shape, long from,
                           long to)
{
    std::vector<float> dataVector(size);
    assert(dataVector.size() == shape.Size());

    m_info = TensorInfo(from, to);
    m_data = std::make_unique<TensorData>(dataVector, std::move(shape));
}

TensorObject::TensorObject(const TensorObject& obj)
{
    if (obj.m_data)
    {
        m_data = std::make_unique<TensorData>(obj.Data(), obj.DataShape());
        m_info = obj.m_info;
    }
}

TensorObject::TensorObject(TensorObject&& obj) noexcept
{
    if (obj.m_data)
    {
        m_data = std::move(obj.m_data);
        m_info = obj.m_info;
    }
}

TensorObject& TensorObject::operator=(const TensorObject& obj)
{
    if (*this == obj)
    {
        return *this;
    }

    if (obj.m_data)
    {
        m_data = std::make_unique<TensorData>(obj.Data(), obj.DataShape());
        m_info = obj.m_info;
    }

    return *this;
}

TensorObject& TensorObject::operator=(TensorObject&& obj) noexcept
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

bool TensorObject::operator==(const TensorObject& obj) const
{
    return Info() == obj.Info();
}

const TensorInfo& TensorObject::Info() const
{
    return m_info;
}

std::vector<float> TensorObject::Data() const
{
    if (!m_data)
    {
        // TODO: Print error log
        return std::vector<float>();
    }

    return m_data->dataVec;
}

std::unique_ptr<TensorData> TensorObject::DataPtr()
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

TensorShape TensorObject::DataShape() const
{
    if (!m_data)
    {
        // TODO: Print error log
        return TensorShape();
    }

    return m_data->shape;
}

void TensorObject::MakeImmutable() const
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