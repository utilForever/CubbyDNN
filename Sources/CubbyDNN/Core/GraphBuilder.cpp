#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Core/GraphBuilder.hpp>
#include <CubbyDNN/Initializer/Constant.hpp>
#include <CubbyDNN/Initializer/Xavier.hpp>
#include <CubbyDNN/Node/Input.hpp>
#include <CubbyDNN/Node/Parameter.hpp>

#include <cassert>

namespace CubbyDNN::Core
{
GraphBuilder::GraphBuilder(Graph* _graph) : graph(_graph)
{
    assert(graph);
}

void GraphBuilder::RegisterStandardNodeType()
{
}

Node::NodeWrapper GraphBuilder::Input(const std::string& nodeName)
{
    return Node::NodeWrapper(graph->CreateNode<Node::Input>(nodeName));
}

Node::NodeWrapper GraphBuilder::Parameter(
    const std::string& nodeName, const Shape& shape,
    Initializer::InitializerWrapper initializer)
{
    return Node::NodeWrapper(
        graph->CreateNode<Node::Parameter>(nodeName, shape, initializer));
}

Initializer::InitializerWrapper GraphBuilder::InitConstant(float constant)
{
    return Initializer::InitializerWrapper(
        graph->CreateInitializer<Initializer::Constant>(constant));
}

Initializer::InitializerWrapper GraphBuilder::InitXavier(
    std::mt19937_64::result_type seed, std::size_t fanIn, std::size_t fanOut)
{
    return Initializer::InitializerWrapper(
        graph->CreateInitializer<Initializer::Xavier>(seed, fanIn, fanOut));
}
}  // namespace CubbyDNN::Core