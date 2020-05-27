#include <CubbyDNN/Optimizer/Momentum.hpp>

#include <utility>

namespace CubbyDNN::Optimizer
{
Momentum::Momentum(float _momentum,
                   std::initializer_list<Node::Parameter*> parameterList)
    : momentum(_momentum), m_parameterList(parameterList)
{
    for (auto* parameter : m_parameterList)
    {
        m_momentumGradientList.emplace_back(
            parameter->EvalShape().Shape().Size());
    }

    for (auto& momentumGradient : m_momentumGradientList)
    {
        momentumGradient.GetSpan().FillZero();
    }
}

Momentum::Momentum(float _momentum, std::vector<Node::Parameter*> parameterList)
    : momentum(_momentum), m_parameterList(std::move(parameterList))
{
    for (auto* parameter : m_parameterList)
    {
        m_momentumGradientList.emplace_back(
            parameter->EvalShape().Shape().Size());
    }

    for (auto& momentumGradient : m_momentumGradientList)
    {
        momentumGradient.GetSpan().FillZero();
    }
}

void Momentum::Reduce(float learningRate, Node::Node* target)
{
    for (auto& momentumGradient : m_momentumGradientList)
    {
        for (auto& gradient : momentumGradient.GetSpan())
        {
            gradient *= momentum;
        }
    }

    auto momentumGradient = m_momentumGradientList.begin();

    for (auto* parameter : m_parameterList)
    {
        momentumGradient->GetSpan().AccumulateFrom(
            -learningRate, parameter->EvalGradient(target).Gradient());
        parameter->GetParameter().AccumulateFrom(momentumGradient->GetSpan());
        parameter->MarkDirty(false);

        ++momentumGradient;
    }
}
}  // namespace CubbyDNN::Optimizer