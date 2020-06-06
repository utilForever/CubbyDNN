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
    Device() = default;
    Device(std::size_t id, DeviceType type, std::string name,
           std::size_t padByteSize = 0);
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

    [[nodiscard]] std::size_t PadSize() const
    {
        return m_padByteSize;
    }

private:
    std::size_t m_id;
    DeviceType m_type;
    std::string m_name;
    std::size_t m_padByteSize;
};
}

#endif
