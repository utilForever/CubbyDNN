#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Core/GraphBuilder.hpp>

namespace CubbyDNN::Core
{
Node::NodeWrapper GraphBuilder::Input(std::string_view nodeName)
{
    return Node::NodeWrapper{ graph->CreateNode<Input>(nodeName) };
}
}  // namespace CubbyDNN