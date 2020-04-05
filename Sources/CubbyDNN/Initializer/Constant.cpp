#include <CubbyDNN/Initializer/Constant.hpp>

namespace CubbyDNN::Initializer
{
Constant::Constant(float constant) : m_constant(constant)
{
    // Do nothing
}

void Constant::operator()(Core::Span<float> span)
{
    (void)span;
}
}  // namespace CubbyDNN::Initializer