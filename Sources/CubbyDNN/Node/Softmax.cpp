#include <CubbyDNN/Node/Softmax.hpp>

namespace CubbyDNN::Node
{
Softmax::Softmax(Core::Graph* graph, std::string_view name,
                 const std::vector<bool>& _groupAxis)
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

void Softmax::EvalShapeInternal()
{
    if (m_inputLogit)
    {
        throw std::runtime_error("No node attached at 'logit'");
    }

    const auto& shape = m_inputLogit.InputNode()->Shape();
    m_shape = shape;

    if (groupAxis.empty() || groupAxis.size() == 1)
    {
        return;
    }

    if (groupAxis.size() != m_shape.Rank())
    {
        throw std::runtime_error(
            "The length of 'group axis' must be equal to the rank of 'logit'");
    }

    std::vector<std::size_t> multipliedShape(1);
    for (std::size_t index = 0, maxIndex = m_shape.Rank() - 1; index < maxIndex;
         ++index)
    {
        multipliedShape.emplace_back(multipliedShape.back() * m_shape[index]);
    }

    m_indexFactorList.clear();

    std::size_t summationSize = m_shape.Size();
    for (std::size_t index = 0, maxIndex = groupAxis.size(); index < maxIndex;
         ++index)
    {
        if (groupAxis[index])
        {
            summationSize /= m_shape[index];
        }
        else
        {
            m_indexFactorList.emplace_back(
                m_shape[index], multipliedShape[index],
                !m_indexFactorList.empty()
                    ? std::get<0>(m_indexFactorList.back()) *
                          std::get<2>(m_indexFactorList.back())
                    : 1);
        }
    }

    m_summation.Resize(summationSize);
}

void Softmax::EvalOutputInternal()
{
    m_inputLogit.InputNode()->EvalOutput();

    const float maxInput = *m_inputLogit.InputNode()->Output().Max();

    if (groupAxis.empty())
    {
        float expSumInv = 0.0f;

        for (auto inputVal : m_inputLogit.InputNode()->Output())
        {
            expSumInv += std::exp(inputVal - maxInput);
        }

        expSumInv = 1.0f / (expSumInv + 1e-4f);

        for (std::size_t index = 0,
                         maxIndex = m_inputLogit.InputNode()->Output().Length();
             index < maxIndex; ++index)
        {
            m_output.GetSpan()[index] =
                expSumInv *
                std::exp(m_inputLogit.InputNode()->Output()[index] - maxInput);
        }

        return;
    }

    if (groupAxis.size() == 1)
    {
        if (groupAxis[0])
        {
            float expSumInv = 0.0f;

            for (auto inputVal : m_inputLogit.InputNode()->Output())
            {
                expSumInv += std::exp(inputVal - maxInput);
            }

            expSumInv = 1.0f / (expSumInv + 1e-4f);

            for (std::size_t
                     index = 0,
                     maxIndex = m_inputLogit.InputNode()->Output().Length();
                 index < maxIndex; ++index)
            {
                m_output.GetSpan()[index] =
                    expSumInv *
                    std::exp(m_inputLogit.InputNode()->Output()[index] -
                             maxInput);
            }
        }
        else
        {
            m_output.GetSpan().FillOne();
        }

        return;
    }

    auto reduceIndex = [this](std::size_t index) {
        std::size_t result = 0;

        for (const auto& indexFactorTuple : m_indexFactorList)
        {
            result += index / std::get<1>(indexFactorTuple) %
                      std::get<0>(indexFactorTuple) *
                      std::get<2>(indexFactorTuple);
        }

        return result;
    };

    for (std::size_t index = 0,
                     maxIndex = m_inputLogit.InputNode()->Output().Length();
         index < maxIndex; ++index)
    {
        m_summation.GetSpan()[reduceIndex(index)] +=
            std::exp(m_inputLogit.InputNode()->Output()[index] - maxInput);
    }

    for (auto& summationVal : m_summation.GetSpan())
    {
        summationVal = 1.0f / summationVal;
    }

    for (std::size_t index = 0,
                     maxIndex = m_inputLogit.InputNode()->Output().Length();
         index < maxIndex; ++index)
    {
        m_output.GetSpan()[index] =
            m_summation.GetSpan()[reduceIndex(index)] *
            std::exp(m_inputLogit.InputNode()->Output()[index] - maxInput);
    }
}
}  // namespace CubbyDNN::Node