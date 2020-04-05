#include <CubbyDNN/Node/Input.hpp>

namespace CubbyDNN::Node
{
Input::Input(Core::Graph* graph, std::string_view name) : Node(graph, name)
{
    // Do nothing
}

const NodeType* Input::Type() const
{
    return graph->nodeTypeManager.Type<Input>();
}

std::string_view Input::TypeName()
{
    return "Input";
}
}  // namespace CubbyDNN::Node