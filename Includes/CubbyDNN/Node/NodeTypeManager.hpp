#ifndef CUBBYDNN_NODE_TYPE_MANAGER_HPP
#define CUBBYDNN_NODE_TYPE_MANAGER_HPP

#include <unordered_map>

namespace CubbyDNN::Node
{
class NodeTypeManager
{
 public:
    const NodeType* Type(const std::string& typeName) const;

    template <typename T>
    const NodeType* Type() const;

 private:
    std::unordered_map<std::string, NodeType> m_nodeTypeMap;
};
}  // namespace CubbyDNN::Node

#include <CubbyDNN/Node/NodeTypeManager-Impl.hpp>

#endif