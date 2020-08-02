#ifndef CUBBYDNN_NODE_TYPE_MANAGER_HPP
#define CUBBYDNN_NODE_TYPE_MANAGER_HPP

#include <unordered_map>

namespace CubbyDNN::Node
{
class NodeTypeManager
{
 public:
    NodeTypeManager();
    ~NodeTypeManager() noexcept = default;

    NodeTypeManager(const NodeTypeManager& rhs) = delete;
    NodeTypeManager(NodeTypeManager&& rhs) noexcept = delete;

    NodeTypeManager& operator=(const NodeTypeManager& rhs) = delete;
    NodeTypeManager& operator=(NodeTypeManager&& rhs) noexcept = delete;

    const NodeType* Type(const std::string& typeName) const;
    template <typename T>
    const NodeType* Type() const;

    template <typename T>
    void RegisterNode();
    template <typename T, typename B>
    void RegisterNode();

 private:
    std::unordered_map<std::string, NodeType> m_nodeTypeMap;
};
}  // namespace CubbyDNN::Node

#include <CubbyDNN/Node/NodeTypeManager-Impl.hpp>

#endif