#include <CubbyDNN/Core/Graph.hpp>

namespace CubbyDNN::Core
{
Graph::Graph() : m_graphBuilder(this)
{
    m_graphBuilder.RegisterStandardNodeType();
}

GraphBuilder& Graph::Builder() noexcept
{
    return m_graphBuilder;
}

std::size_t Graph::NodeCount(const Node::NodeType* nodeType) const
{
    return m_nodeTypeMap.count(nodeType);
}

Node::Node* Graph::Node(const std::string& nodeName) const
{
    const auto iter = this->m_nodeMap.find(nodeName);

    return iter == m_nodeMap.cend() ? nullptr : iter->second.get();
}
}  // namespace CubbyDNN::Core