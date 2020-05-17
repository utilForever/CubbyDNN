#include <CubbyDNN/Node/SoftmaxCE.hpp>

namespace CubbyDNN::Node
{
SoftmaxCE::SoftmaxCE(Core::Graph* graph, std::string_view name)
    : Node(graph, name),
      m_inputLabel(this, "label", [this](const auto* dy) { (void)dy; }),
      m_inputProb(this, "prob", [this](const auto* dy) { (void)dy; })
{
    m_nodeInputMap["label"] = &m_inputLabel;
    m_nodeInputMap["prob"] = &m_inputProb;
}

const NodeType* SoftmaxCE::Type() const
{
    return graph->nodeTypeManager.Type<SoftmaxCE>();
}

std::string_view SoftmaxCE::TypeName()
{
    return "SoftmaxCE";
}
}  // namespace CubbyDNN::Node