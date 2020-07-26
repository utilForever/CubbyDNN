#include <CubbyDNN/Node/Node.hpp>
#include <CubbyDNN/Node/NodeType.hpp>
#include <CubbyDNN/Node/NodeTypeManager.hpp>

namespace CubbyDNN::Node
{
NodeTypeManager::NodeTypeManager()
{
    auto typeName = Node::TypeName();
    m_nodeTypeMap.emplace(std::piecewise_construct,
                          std::forward_as_tuple(std::string(typeName)),
                          std::forward_as_tuple(nullptr, typeName));
}

const NodeType* NodeTypeManager::Type(const std::string& typeName) const
{
    const auto iter = m_nodeTypeMap.find(typeName);

    return iter == m_nodeTypeMap.cend() ? nullptr : &iter->second;
}
}  // namespace CubbyDNN::Node
