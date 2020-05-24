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

void SoftmaxCE::EvalShapeInternal()
{
    if (!m_inputLabel)
    {
        throw std::runtime_error("No node attached at 'label'");
    }

    if (!m_inputProb)
    {
        throw std::runtime_error("No node attached at 'prob'");
    }

    if (m_inputLabel.InputNode()->EvalShape().Shape() !=
        m_inputProb.InputNode()->EvalShape().Shape())
    {
        throw std::runtime_error(
            "The shape of 'label' and 'prob' must be equal");
    }

    m_shape = { 1 };
}

void SoftmaxCE::EvalOutputInternal()
{
    m_inputLabel.InputNode()->EvalOutput();
    m_inputProb.InputNode()->EvalOutput();

    m_output.GetSpan()[0] = 0.0f;

    for (std::size_t index = 0,
                     maxIndex = m_inputLabel.InputNode()->Shape().Size();
         index < maxIndex; ++index)
    {
        m_output.GetSpan()[0] +=
            m_inputLabel.InputNode()->Output()[index] *
            std::log(m_inputProb.InputNode()->Output()[index] + 1e-4f);
    }

    m_output.GetSpan()[0] /=
        -static_cast<float>(m_inputLabel.InputNode()->Shape()[1]);
}
}  // namespace CubbyDNN::Node