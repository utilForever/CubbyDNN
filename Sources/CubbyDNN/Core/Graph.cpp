#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Node/Input.hpp>

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

void Graph::Feed(const std::vector<std::tuple<std::string, Shape, Span<float>>>&
                     feedDataList) const
{
    for (const auto& feedData : feedDataList)
    {
        Node<Node::Input>(std::get<0>(feedData))
            ->Feed(std::get<1>(feedData), std::get<2>(feedData));
    }
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