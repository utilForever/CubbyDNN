#include <CubbyDNN/Node/Parameter.hpp>

#include <utility>

namespace CubbyDNN::Node
{
Parameter::Parameter(Core::Graph* _graph, std::string_view _name,
                     Core::Shape _shape, Initializer::Initializer* _initializer)
    : Node(_graph, _name),
      parameterShape(std::move(_shape)),
      initializer(_initializer)

{
    m_parameter.Resize(EvalShape().Shape().Size());
    (*initializer)(m_parameter.GetSpan());
}

Core::Span<float> Parameter::GetParameter() const noexcept
{
    return m_parameter.GetSpan();
}
}  // namespace CubbyDNN::Node