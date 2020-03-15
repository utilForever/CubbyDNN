// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Tensors/TensorInfo.hpp>

namespace CubbyDNN
{
std::map<NumberSystem, std::size_t> TensorInfo::UnitByteSizeMap = {
    { NumberSystem::Float, sizeof(float) },
    { NumberSystem::Int, sizeof(int) },
};

TensorInfo::TensorInfo(Shape shape,
                       NumberSystem numberSystem)
    : m_shape(std::move(shape)),
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

std::size_t TensorInfo::GetSize() const noexcept
{
    return m_shape.TotalSize();
}

std::size_t TensorInfo::GetByteSize() const noexcept
{
    return m_unitByteSize * GetSize();
}

const Shape& TensorInfo::GetShape() const noexcept
{
    return m_shape;
}
} // namespace CubbyDNN
