#include <CubbyDNN/Node/NodeType.hpp>
#include <CubbyDNN/Node/NodeTypeManager.hpp>

namespace CubbyDNN::Node
{
const NodeType* NodeTypeManager::Type(const std::string& typeName) const
{
    const auto iter = m_nodeTypeMap.find(typeName);

    return iter == m_nodeTypeMap.cend() ? nullptr : &iter->second;
}
}  // namespace CubbyDNN::Node
