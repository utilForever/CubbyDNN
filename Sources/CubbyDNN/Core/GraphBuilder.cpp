#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Core/GraphBuilder.hpp>
#include <CubbyDNN/Initializer/Constant.hpp>
#include <CubbyDNN/Initializer/Xavier.hpp>
#include <CubbyDNN/Node/Dense.hpp>
#include <CubbyDNN/Node/Input.hpp>
#include <CubbyDNN/Node/Parameter.hpp>
#include <CubbyDNN/Node/ReLU.hpp>
#include <CubbyDNN/Node/Softmax.hpp>
#include <CubbyDNN/Node/SoftmaxCE.hpp>

#include <cassert>
#include <sstream>

template <typename T>
constexpr std::string GetDefaultName(CubbyDNN::Core::Graph* graph)
{
    return (std::ostringstream{} << T::TypeName() << graph->NodeCount<T>())
        .str();
}

namespace CubbyDNN::Core
{
GraphBuilder::GraphBuilder(Graph* _graph) : graph(_graph)
{
    assert(graph);
}

void GraphBuilder::RegisterStandardNodeType()
{
    graph->nodeTypeManager.RegisterNode<Node::Input>();
    graph->nodeTypeManager.RegisterNode<Node::Parameter>();

    graph->nodeTypeManager.RegisterNode<Node::ReLU>();
    graph->nodeTypeManager.RegisterNode<Node::Softmax>();

    graph->nodeTypeManager.RegisterNode<Node::Dense>();

    graph->nodeTypeManager.RegisterNode<Node::SoftmaxCE>();
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

Node::NodeWrapper GraphBuilder::ReLU(Node::NodeWrapper logit, float alpha)
{
    Node::NodeWrapper node(graph->CreateNode<Node::ReLU>(
        GetDefaultName<Node::ReLU>(graph), alpha));

    node["logit"]->Attach(logit);

    return node;
}

Node::NodeWrapper GraphBuilder::Softmax(Node::NodeWrapper logit,
                                        const std::vector<bool>& reduceAxis)
{
    Node::NodeWrapper node(graph->CreateNode<Node::Softmax>(
        GetDefaultName<Node::Softmax>(graph), reduceAxis));

    node["logit"]->Attach(logit);

    return node;
}

Node::NodeWrapper GraphBuilder::Dense(Node::NodeWrapper input,
                                      Node::NodeWrapper weight,
                                      Node::NodeWrapper bias)
{
    Node::NodeWrapper node(
        graph->CreateNode<Node::Dense>(GetDefaultName<Node::Dense>(graph)));

    node["input"]->Attach(input);
    node["weight"]->Attach(weight);
    node["bias"]->Attach(bias);

    return node;
}

Node::NodeWrapper GraphBuilder::SoftmaxCE(Node::NodeWrapper label,
                                          Node::NodeWrapper prob)
{
    Node::NodeWrapper node(graph->CreateNode<Node::SoftmaxCE>(
        GetDefaultName<Node::SoftmaxCE>(graph)));

    node["label"]->Attach(label);
    node["prob"]->Attach(prob);

    return node;
}

Initializer::InitializerWrapper GraphBuilder::InitXavier(
    std::mt19937_64::result_type seed, std::size_t fanIn, std::size_t fanOut)
{
    return Initializer::InitializerWrapper(
        graph->CreateInitializer<Initializer::Xavier>(seed, fanIn, fanOut));
}
}  // namespace CubbyDNN::Core