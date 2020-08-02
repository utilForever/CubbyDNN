#include <CubbyDNN/Node/SoftmaxCE.hpp>

namespace CubbyDNN::Node
{
SoftmaxCE::SoftmaxCE(Core::Graph* graph, std::string_view name)
    : Node(graph, name),
      m_inputLabel(this, "label",
                   [this](const auto* dy) { BackwardOpLabel(dy); }),
      m_inputProb(this, "prob", [this](const auto* dy) { BackwardOpProb(dy); })
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

void SoftmaxCE::BackwardOpLabel(const Node* dy)
{
    m_inputProb.InputNode()->EvalOutput();
    EvalGradient(dy);

    const float factor =
        -m_gradient.GetSpan()[0] / m_inputProb.InputNode()->Shape()[1];

    for (std::size_t index = 0,
                     maxIndex = m_inputProb.InputNode()->Gradient().Length();
         index < maxIndex; ++index)
    {
        m_inputLabel.InputNode()->Gradient()[index] +=
            factor * std::log(m_inputProb.InputNode()->Output()[index] + 1e-4f);
    }
}

void SoftmaxCE::BackwardOpProb(const Node* dy)
{
    m_inputLabel.InputNode()->EvalOutput();
    m_inputProb.InputNode()->EvalOutput();
    EvalGradient(dy);

    const float factor =
        -m_gradient.GetSpan()[0] / m_inputProb.InputNode()->Shape()[1];

    for (std::size_t index = 0,
                     maxIndex = m_inputProb.InputNode()->Gradient().Length();
         index < maxIndex; ++index)
    {
        m_inputProb.InputNode()->Gradient()[index] +=
            factor * m_inputLabel.InputNode()->Output()[index] /
            (m_inputProb.InputNode()->Output()[index] + 1e-4f);
    }
}
}  // namespace CubbyDNN::Node