#ifndef CUBBYDNN_NODE_HPP
#define CUBBYDNN_NODE_HPP

#include <CubbyDNN/Node/NodeInput.hpp>
#include <CubbyDNN/Node/NodeType.hpp>

#include <string_view>
#include <unordered_set>

namespace CubbyDNN::Core
{
class Graph;
}

namespace CubbyDNN::Node
{
class Node
{
 public:
    friend NodeInput;

    Node(Core::Graph* _graph, std::string_view _name);

    virtual const NodeType* Type() const;
    static std::string_view TypeName();

    Core::Graph* const graph;
    const std::string name;

 private:
    std::vector<NodeInput*> m_revNodeInputList;
    std::unordered_set<Node*> m_deps;
    std::unordered_set<Node*> m_revDeps;
    std::unordered_map<std::string, NodeInput*> m_nodeInputMap;
};
}  // namespace CubbyDNN::Node

#endif