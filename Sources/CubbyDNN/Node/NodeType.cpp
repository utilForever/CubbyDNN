#include <CubbyDNN/Node/NodeType.hpp>

namespace CubbyDNN
{
NodeType::NodeType(const NodeType* _baseType, std::string_view _typeName)
    : baseType(_baseType), typeName(_typeName)
{
    // Do nothing
}

bool NodeType::operator==(const NodeType& rhs) const
{
    return typeName == rhs.typeName;
}
}  // namespace CubbyDNN