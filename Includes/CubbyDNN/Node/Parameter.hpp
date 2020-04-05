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

    const Core::Shape parameterShape;
    Initializer::Initializer* const initializer;
};
}  // namespace CubbyDNN::Node

#endif