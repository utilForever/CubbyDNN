#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Node/Node.hpp>

namespace CubbyDNN::Node
{
Node::Node(Core::Graph* _graph, std::string_view _name)
    : graph(_graph), name(_name)
{
    // Do nothing
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