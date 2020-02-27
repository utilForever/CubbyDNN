// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cstdio>
#include <cubbydnn/Tensors/TensorInfo.hpp>

namespace CubbyDNN
{
std::map<NumberSystem, size_t> TensorInfo::UnitByteSizeMap = {
    { NumberSystem::Float16, 16 }, { NumberSystem::Float32, 32 },
    { NumberSystem::Float64, 64 }, { NumberSystem::Int8, 8 },
    { NumberSystem::Int16, 16 }, { NumberSystem::Int32, 32 },
    { NumberSystem::Int64, 64 },
};

TensorInfo::TensorInfo(const Shape& shape,
                       NumberSystem numberSystem)
    : m_shape(shape),
      m_unitByteSize(UnitByteSizeMap.at(numberSystem)),
      m_numberSystem(numberSystem)
{
}

bool TensorInfo::operator==(const TensorInfo& tensorInfo) const
{
    return (m_shape == tensorInfo.m_shape &&
            m_numberSystem == tensorInfo.m_numberSystem &&
            m_unitByteSize == tensorInfo.m_unitByteSize);
}

bool TensorInfo::operator!=(const TensorInfo& tensorInfo) const
{
    return !(*this == tensorInfo);
}

size_t TensorInfo::GetSize() const noexcept
{
    return m_shape.GetTotalSize();
}

size_t TensorInfo::GetByteSize() const noexcept
{
    return m_unitByteSize * GetSize();
}

const Shape& TensorInfo::GetShape() const noexcept
{
    return m_shape;
}
} // namespace CubbyDNN
