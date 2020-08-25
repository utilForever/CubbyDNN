// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_DEVICE_HPP
#define TAKION_DEVICE_HPP
#include <string>

namespace Takion::Compute
{
enum class DeviceType
{
    CPU,
    GPU,
    HYBRID
};

class Device
{
public:
    Device() = default;

    Device(int id, DeviceType type, std::string name);
    ~Device() = default;

    Device(const Device& device) = default;
    Device(Device&& device) noexcept = default;
    Device& operator=(const Device& device) = default;
    Device& operator=(Device&& device) noexcept = default;

    bool operator==(const Device& device) const;
    bool operator!=(const Device& device) const;

    [[nodiscard]] DeviceType Type() const
    {
        return m_type;
    }

    [[nodiscard]] std::string Name() const
    {
        return m_name;
    }

    [[nodiscard]] std::size_t PadByteSize() const
    {
        return m_padByteSize;
    }

private:
    int m_id = -1;
    DeviceType m_type = DeviceType::CPU;
    std::string m_name = "Undefined";
    std::size_t m_padByteSize = 0;
};
}

#endif
