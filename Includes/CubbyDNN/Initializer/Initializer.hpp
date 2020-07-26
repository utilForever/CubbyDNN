#ifndef CUBBYDNN_INITIALIZER_HPP
#define CUBBYDNN_INITIALIZER_HPP

#include <CubbyDNN/Core/Span.hpp>

namespace CubbyDNN::Initializer
{
class Initializer
{
 public:
    Initializer() = default;
    virtual ~Initializer() noexcept = default;

    Initializer(const Initializer& rhs) = default;
    Initializer(Initializer&& rhs) noexcept = default;

    Initializer& operator=(const Initializer& rhs) = default;
    Initializer& operator=(Initializer&& rhs) noexcept = default;

    virtual void operator()(Core::Span<float> span) = 0;
};
}  // namespace CubbyDNN::Initializer

#endif