// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Tensors/Tensor.hpp>

#include <utility>

namespace CubbyDNN
{
Tensor::Tensor(TensorShape shape, long prevOpID, bool isMutable) noexcept
    : m_shape(std::move(shape)), m_prevOpID(prevOpID), m_isMutable(isMutable)
{
    // Do nothing
}

const TensorShape& Tensor::Shape() const noexcept
{
    return m_shape;
}

std::size_t Tensor::DataSize() const noexcept
{
    return m_shape.Size();
}

long Tensor::PrevOpID() const noexcept
{
    return m_prevOpID;
}

void Tensor::AddOp(long nextOpID)
{
    m_nextOps.emplace_back(nextOpID);
}

bool Tensor::IsValid() const noexcept
{
    return !m_shape.IsEmpty();
}

bool Tensor::IsMutable() const noexcept
{
    return m_isMutable;
}

void Tensor::MakeImmutable() noexcept
{
    m_isMutable = false;
}
}  // namespace CubbyDNN