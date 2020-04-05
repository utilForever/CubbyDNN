#include <CubbyDNN/Core/Graph.hpp>

namespace CubbyDNN::Core
{
GraphBuilder& Graph::Builder() noexcept
{
    return m_graphBuilder;
}

Node::Node* Graph::Node(const std::string& nodeName) const
{
    const auto iter = this->m_nodeMap.find(nodeName);

    return iter == m_nodeMap.cend() ? nullptr : iter->second.get();
}
}  // namespace CubbyDNN::Core