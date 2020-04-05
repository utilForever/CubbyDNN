#ifndef CUBBYDNN_GRAPH_HPP
#define CUBBYDNN_GRAPH_HPP

#include <CubbyDNN/Core/GraphBuilder.hpp>
#include <CubbyDNN/Node/Node.hpp>
#include <CubbyDNN/Node/NodeType.hpp>
#include <CubbyDNN/Node/NodeTypeManager.hpp>

#include <memory>
#include <unordered_map>

namespace CubbyDNN::Core
{
class Graph
{
 public:
    GraphBuilder& Builder() noexcept;

    Node::Node* Node(const std::string& nodeName) const;

    template <typename T>
    T* Node(const std::string& nodeName) const;

    template <typename T>
    T* CreateNode(std::string_view nodeName);

    Node::NodeTypeManager nodeTypeManager;

 private:
    GraphBuilder m_graphBuilder;

    std::unordered_map<std::string, std::unique_ptr<Node::Node>> m_nodeMap;
    std::unordered_multimap<const Node::NodeType*, Node::Node*> m_nodeTypeMap;
};
}  // namespace CubbyDNN::Core

#include <CubbyDNN/Core/Graph-Impl.hpp>

#endif