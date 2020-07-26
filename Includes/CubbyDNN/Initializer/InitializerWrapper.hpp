#ifndef CUBBYDNN_INITIALIZER_WRAPPER_HPP
#define CUBBYDNN_INITIALIZER_WRAPPER_HPP

#include <CubbyDNN/Initializer/Initializer.hpp>

namespace CubbyDNN::Initializer
{
class InitializerWrapper
{
 public:
    InitializerWrapper(Initializer* _initializer);

    operator Initializer*() noexcept;
    operator const Initializer*() const noexcept;

    Initializer* const initializer;
};
}  // namespace CubbyDNN::Initializer

#endif