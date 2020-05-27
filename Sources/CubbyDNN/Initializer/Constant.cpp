#include <CubbyDNN/Initializer/Constant.hpp>

namespace CubbyDNN::Initializer
{
Constant::Constant(float constant) : m_constant(constant)
{
    // Do nothing
}

void Constant::operator()(Core::Span<float> span)
{
    span.FillScalar(m_constant);
}
}  // namespace CubbyDNN::Initializer