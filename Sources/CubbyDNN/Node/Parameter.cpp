#include <CubbyDNN/Node/Parameter.hpp>

namespace CubbyDNN::Node
{
Parameter::Parameter(Core::Graph* _graph, std::string_view _name,
                     Core::Shape _shape, Initializer::Initializer* _initializer)
    : Node(_graph, _name), parameterShape(_shape), initializer(_initializer)

{
    // Do nothing
}
}  // namespace CubbyDNN::Node