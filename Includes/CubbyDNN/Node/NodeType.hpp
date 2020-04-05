#ifndef CUBBYDNN_NODE_TYPE_HPP
#define CUBBYDNN_NODE_TYPE_HPP

#include <string>
#include <string_view>

namespace CubbyDNN::Node
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

    bool IsBaseOf(const NodeType* _derivedType) const;
    bool IsDerivedFrom(const NodeType* _baseType) const;
    bool IsExactlyBaseOf(const NodeType* _derivedType) const;
    bool IsExactlyDerivedFrom(const NodeType* _baseType) const;

    const NodeType* baseType;
    const std::string typeName;

 private:
    static bool IsBaseOf(const NodeType* _baseType,
                         const NodeType* _derivedType);
    static bool IsExactlyBaseOf(const NodeType* _baseType,
                                const NodeType* _derivedType);
};
}  // namespace CubbyDNN

#endif