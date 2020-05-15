// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_DEVICE_HPP
#define CUBBYDNN_DEVICE_HPP
#include <string>

namespace CubbyDNN::Compute
{
enum class DeviceType
{
    Cpu,
    Blaze,
    Cuda,
};

class Device
{
public:
    Device(std::size_t id, DeviceType type, std::string name);
    ~Device() = default;

    Device(const Device& device) = default;
    Device(Device&& device) noexcept = default;
    Device& operator=(const Device& device) = default;
    Device& operator=(Device&& device) noexcept = default;

    bool operator==(const Device& device)
    {
        return m_id == device.m_id
               && type == device.type
               && m_name == device.m_name;
    }

private:
    std::size_t m_id;
    DeviceType type;
    std::string m_name;
};
}

#endif
