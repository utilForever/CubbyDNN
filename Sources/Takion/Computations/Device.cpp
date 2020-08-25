// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Takion/Computations/Device.hpp>
#include <stdexcept>

namespace Takion::Compute
{
Device::Device(int id, DeviceType type, std::string name)
    : m_id(id),
      m_type(type),
      m_name(std::move(name))
{
    if (type == DeviceType::CPU)
        m_padByteSize = 32;
    else if (type == DeviceType::GPU)
        m_padByteSize = 1;
}

bool Device::operator==(const Device& device) const
{
    return m_id == device.m_id && m_type == device.m_type &&
           m_name == device.m_name && m_padByteSize == device.m_padByteSize;
}

bool Device::operator!=(const Device& device) const
{
    return !(*this == device);
}
}
