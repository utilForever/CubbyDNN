#ifndef CUBBYDNN_INPUT_HPP
#define CUBBYDNN_INPUT_HPP

#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Node/Node.hpp>

namespace CubbyDNN::Node
{
class Input : public Node
{
 public:
    Input(Core::Graph* graph, std::string_view name);

    const NodeType* Type() const override;
    static std::string_view TypeName();
};
}  // namespace CubbyDNN::Node

#endif