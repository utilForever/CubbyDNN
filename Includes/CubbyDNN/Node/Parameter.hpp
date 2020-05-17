#ifndef CUBBYDNN_PARAMETER_HPP
#define CUBBYDNN_PARAMETER_HPP

#include <CubbyDNN/Core/Shape.hpp>
#include <CubbyDNN/Initializer/Initializer.hpp>
#include <CubbyDNN/Node/Node.hpp>

namespace CubbyDNN::Node
{
class Parameter : public Node
{
 public:
    Parameter(Core::Graph* _graph, std::string_view _name, Core::Shape _shape,
              Initializer::Initializer* _initializer);
    Parameter(const Parameter& rhs) = delete;
    Parameter(Parameter&& rhs) noexcept = delete;

    virtual ~Parameter() noexcept = default;

    Parameter& operator=(const Parameter& rhs) = delete;
    Parameter& operator=(Parameter&& rhs) noexcept = delete;

    Core::Span<float> GetParameter() const noexcept;

    const Core::Shape parameterShape;
    Initializer::Initializer* const initializer;

private:
    Core::Memory<float> m_parameter;
};
}  // namespace CubbyDNN::Node

#endif