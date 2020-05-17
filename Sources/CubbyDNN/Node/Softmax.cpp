#include <CubbyDNN/Node/Softmax.hpp>

namespace CubbyDNN::Node
{
Softmax::Softmax(Core::Graph* graph, std::string_view name, const std::vector<bool>& _groupAxis)
    : Node(graph, name),
      groupAxis(_groupAxis),
      m_inputLogit(this, "logit", [this](const auto* dy) { (void)dy; })
{
    m_nodeInputMap["logit"] = &m_inputLogit;
}

const NodeType* Softmax::Type() const
{
    return graph->nodeTypeManager.Type<Softmax>();
}

std::string_view Softmax::TypeName()
{
    return "Softmax";
}
}  // namespace CubbyDNN::Node