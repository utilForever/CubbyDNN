#include <CubbyDNN/Node/ReLU.hpp>

namespace CubbyDNN::Node
{
ReLU::ReLU(Core::Graph* graph, std::string_view name, float _alpha)
    : Node(graph, name),
      alpha(_alpha),
      m_inputLogit(this, "logit", [this](const auto* dy) { (void)dy; })
{
    m_nodeInputMap["logit"] = &m_inputLogit;
}

const NodeType* ReLU::Type() const
{
    return graph->nodeTypeManager.Type<ReLU>();
}

std::string_view ReLU::TypeName()
{
    return "ReLU";
}

void ReLU::EvalShapeInternal()
{
    if (!m_inputLogit)
    {
        throw std::runtime_error("No node attached at 'logit'");
    }

    m_shape = m_inputLogit.InputNode()->Shape();
}

void ReLU::EvalOutputInternal()
{
    m_inputLogit.InputNode()->EvalOutput();

    const auto rectify = [](float value, float alpha) {
        return value < 0.0f ? alpha * value : value;
    };

    for (std::size_t index = 0, maxIndex = m_output.Size(); index < maxIndex;
         ++index)
    {
        m_output.GetSpan()[index] =
            rectify(m_inputLogit.InputNode()->Output()[index], alpha);
    }
}
}  // namespace CubbyDNN::Node