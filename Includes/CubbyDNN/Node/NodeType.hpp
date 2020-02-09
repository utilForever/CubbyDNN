#ifndef CUBBYDNN_NODE_TYPE_HPP
#define CUBBYDNN_NODE_TYPE_HPP

#include <string>
#include <string_view>

namespace CubbyDNN
{
class NodeType
{
 public:
    NodeType(const NodeType* _baseType, std::string_view _typeName);
    ~NodeType() noexcept = default;

    NodeType(const NodeType& rhs) = delete;
    NodeType(NodeType&& rhs) = delete;
    NodeType& operator=(const NodeType& rhs) = delete;
    NodeType& operator=(NodeType&& rhs) = delete;

    bool operator==(const NodeType& rhs) const;

    const NodeType* baseType;
    const std::string typeName;
};
}  // namespace CubbyDNN

#endif