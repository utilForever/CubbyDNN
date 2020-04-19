#include <CubbyDNN/Node/Dense.hpp>

namespace CubbyDNN::Node
{
Dense::Dense(Core::Graph* graph, std::string_view name)
    : Node(graph, name),
      m_input(this, "input", [this](const auto* dy) { (void)dy; }),
      m_inputWeight(this, "weight", [this](const auto* dy) { (void)dy; }),
      m_inputBias(this, "bias", [this](const auto* dy) { (void)dy; })
{
    m_nodeInputMap["input"] = &m_input;
    m_nodeInputMap["weight"] = &m_inputWeight;
    m_nodeInputMap["bias"] = &m_inputBias;
}

const NodeType* Dense::Type() const
{
    return graph->nodeTypeManager.Type<Dense>();
}

std::string_view Dense::TypeName()
{
    return "Dense";
}
}  // namespace CubbyDNN::Node