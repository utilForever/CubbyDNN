#ifndef CUBBYDNN_GRAPH_HPP
#define CUBBYDNN_GRAPH_HPP

#include <CubbyDNN/Core/GraphBuilder.hpp>
#include <CubbyDNN/Node/Node.hpp>
#include <CubbyDNN/Node/NodeType.hpp>
#include <CubbyDNN/Node/NodeTypeManager.hpp>

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace CubbyDNN::Core
{
class Graph
{
 public:
    Graph();

    GraphBuilder& Builder() noexcept;

    std::size_t NodeCount(const Node::NodeType* nodeType) const;

    Node::Node* Node(const std::string& nodeName) const;

    template <typename T>
    std::size_t NodeCount() const;

    template <typename T>
    T* Node(const std::string& nodeName) const;

    template <typename T, typename... P>
    T* CreateNode(const std::string& nodeName, P&&... params);

    template <typename T, typename... P>
    T* CreateInitializer(P&&... params);

    Node::NodeTypeManager nodeTypeManager;

 private:
    GraphBuilder m_graphBuilder;

    std::unordered_map<std::string, std::unique_ptr<Node::Node>> m_nodeMap;
    std::unordered_multimap<const Node::NodeType*, Node::Node*> m_nodeTypeMap;
    std::unordered_set<std::unique_ptr<Initializer::Initializer>>
        m_intializerSet;
};
}  // namespace CubbyDNN::Core

#include <CubbyDNN/Core/Graph-Impl.hpp>

#endif