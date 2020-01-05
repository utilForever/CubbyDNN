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
    { NumberSystem::Int16, 16 },   { NumberSystem::Int32, 32 },
    { NumberSystem::Int64, 64 },
};

TensorInfo::TensorInfo(std::vector<size_t> shape, NumberSystem numberSystem, bool isMutable)
    : m_getTotalByteSize([](const std::vector<size_t>& shape) {
          size_t totalSize = 1;
          for (auto elem : shape)
          {
              totalSize *= elem;
          }
          return totalSize;
      }),
      m_shape(std::move(shape)),
      m_totalSize(m_getTotalByteSize(shape)),
      m_unitByteSize(UnitByteSizeMap.at(numberSystem)),
      m_numberSystem(numberSystem),
      m_isMutable(isMutable)
{
}

bool TensorInfo::operator==(const TensorInfo& shape) const
{
    return (m_shape == shape.m_shape &&
            m_numberSystem == shape.m_numberSystem &&
            m_unitByteSize == shape.m_unitByteSize);
}

bool TensorInfo::operator!=(const TensorInfo& shape) const
{
    return !(*this == shape);
}

size_t TensorInfo::Size() const noexcept
{
    return m_totalSize;
}

size_t TensorInfo::ByteSize() const noexcept
{
    return m_unitByteSize * Size();
}

bool TensorInfo::IsEmpty() const noexcept
{
    return m_shape.empty();
}

const std::vector<size_t>& TensorInfo::GetShape() const noexcept
{
    return m_shape;
}

}  // namespace CubbyDNN