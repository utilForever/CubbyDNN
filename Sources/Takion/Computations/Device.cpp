// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Takion/Computations/Device.hpp>
#include <stdexcept>

namespace takion::Compute
{
Device::Device(int id, DeviceType type, std::string name,
               std::size_t padByteSize)
    : m_id(id),
      m_type(type),
      m_name(std::move(name)),
      m_padByteSize(padByteSize)
{
    if (type != DeviceType::Blaze && padByteSize > 0)
        throw std::invalid_argument(
            "Only blaze device can have padded tensors");
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
