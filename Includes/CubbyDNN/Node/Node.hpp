#ifndef CUBBYDNN_NODE_HPP
#define CUBBYDNN_NODE_HPP

#include <CubbyDNN/Node/NodeType.hpp>

#include <string_view>

namespace CubbyDNN::Core
{
class Graph;
}

namespace CubbyDNN::Node
{
class Node
{
 public:
    Node(Core::Graph* _graph, std::string_view _name);

    virtual const NodeType* Type() const;
    static std::string_view TypeName();

    Core::Graph* const graph;
    const std::string name;
};
}  // namespace CubbyDNN::Node

#endif