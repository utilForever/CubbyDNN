#include <CubbyDNN/Node/Dense.hpp>

#include <CubbyDNN/Compute/GEMM.hpp>

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

void Dense::EvalShapeInternal()
{
    if (!m_input)
    {
        throw std::runtime_error("No node attached at 'input'");
    }

    if (m_input.InputNode()->Shape().Rank() != 2)
    {
        throw std::runtime_error("The rank of 'input' must be 2");
    }

    if (m_inputWeight.InputNode()->Shape().Rank() != 2)
    {
        throw std::runtime_error("The rank of 'weight' must be 2");
    }

    if (m_inputBias && m_inputBias.InputNode()->Shape().Rank() != 1)
    {
        throw std::runtime_error("The rank of 'bias' must be 1");
    }

    if (m_input.InputNode()->Shape()[0] !=
        m_inputWeight.InputNode()->Shape()[1])
    {
        throw std::runtime_error(
            "The shape of 'input' and 'weight' is not compatible");
    }

    if (m_inputBias && m_inputWeight.InputNode()->Shape()[0] !=
                           m_inputBias.InputNode()->Shape()[0])
    {
        throw std::runtime_error(
            "The shape of 'weight' and 'bias' is not compatible");
    }

    m_shape = { m_inputWeight.InputNode()->Shape()[0],
                m_input.InputNode()->Shape()[1] };
}

void Dense::EvalOutputInternal()
{
    if (m_inputBias)
    {
        m_inputBias.InputNode()->EvalOutput();

        for (std::size_t index = 0, maxIndex = m_shape[1], width = m_shape[0];
             index < maxIndex; ++index)
        {
            m_output.GetSpan()
                .SubSpan(index * width)
                .CopyFrom(m_inputBias.InputNode()->Output());
        }
    }
    else
    {
        m_output.GetSpan().FillZero();
    }

    Compute::GEMM::MultiplyAdd(
        m_input.InputNode()->Shape()[0], m_shape[1], m_shape[0],
        m_input.InputNode()->EvalOutput().Output(),
        m_inputWeight.InputNode()->EvalOutput().Output(), m_output.GetSpan());
}
}  // namespace CubbyDNN::Node