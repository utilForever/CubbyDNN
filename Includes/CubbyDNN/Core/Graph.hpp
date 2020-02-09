#ifndef CUBBYDNN_GRAPH_HPP
#define CUBBYDNN_GRAPH_HPP

#include <CubbyDNN/Node/Node.hpp>
#include <CubbyDNN/Node/NodeType.hpp>

#include <memory>
#include <unordered_map>

namespace CubbyDNN
{
class Graph
{
 private:
    std::unordered_map<std::string, std::unique_ptr<Node>> m_nodeMap;
    std::unordered_multimap<const NodeType*, Node*> m_nodeTypeMap;
};
}  // namespace CubbyDNN

#endif