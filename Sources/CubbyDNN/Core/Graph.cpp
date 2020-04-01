#include <CubbyDNN/Core/Graph.hpp>

namespace CubbyDNN
{
GraphBuilder& Graph::Builder() noexcept
{
    return m_graphBuilder;
}
}  // namespace CubbyDNN