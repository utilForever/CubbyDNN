#include <CubbyDNN/Initializer/InitializerWrapper.hpp>

namespace CubbyDNN::Initializer
{
InitializerWrapper::InitializerWrapper(Initializer* _initializer)
    : initializer(_initializer)
{
    // Do nothing
}

InitializerWrapper::operator Initializer*() noexcept
{
    return initializer;
}

InitializerWrapper::operator const Initializer*() const noexcept
{
    return initializer;
}
}  // namespace CubbyDNN::Initializer