// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Computations/Device.hpp>
#include <stdexcept>

namespace CubbyDNN::Compute
{
Device::Device(std::size_t id, DeviceType type, std::string name,
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
           m_name == device.m_name;
}

bool Device::operator!=(const Device& device) const
{
    return !(*this == device);
}
}
