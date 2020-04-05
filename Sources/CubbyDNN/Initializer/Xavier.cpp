#include <CubbyDNN/Initializer/Xavier.hpp>

namespace CubbyDNN::Initializer
{
Xavier::Xavier(std::mt19937_64::result_type seed, std::size_t fanIn,
               [[maybe_unused]] std::size_t fanOut)
    : m_engine(seed), m_dist(0.0f, std::sqrt(2.0f / fanIn))
{
    // Do nothing
}

void Xavier::operator()(Core::Span<float> span)
{
    (void)span;
}
}  // namespace CubbyDNN::Initializer