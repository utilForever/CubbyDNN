#include <CubbyDNN/Node/ReLU.hpp>

namespace CubbyDNN::Node
{
ReLU::ReLU(Core::Graph* graph, std::string_view name, float _alpha)
    : Node(graph, name),
      alpha(_alpha),
      m_inputLogit(this, "logit", [this](const auto* dy) { (void)dy; })
{
    m_nodeInputMap["input"] = &m_inputLogit;
}

const NodeType* ReLU::Type() const
{
    return graph->nodeTypeManager.Type<ReLU>();
}

std::string_view ReLU::TypeName()
{
    return "ReLU";
}
}  // namespace CubbyDNN::Node