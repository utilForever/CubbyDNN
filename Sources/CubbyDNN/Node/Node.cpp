#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Node/Node.hpp>

namespace CubbyDNN::Node
{
Node::Node(Core::Graph* _graph, std::string_view _name)
    : graph(_graph), name(_name)
{
    // Do nothing
}

NodeInput* Node::operator[](const std::string& inputName)
{
    const auto index(m_nodeInputMap.find(inputName));
    return index == m_nodeInputMap.cend() ? nullptr : index->second;
}

const NodeType* Node::Type() const
{
    return graph->nodeTypeManager.Type<Node>();
}

std::string_view Node::TypeName()
{
    return "Node";
}
}  // namespace CubbyDNN::Node