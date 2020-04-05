#ifndef CUBBYDNN_GRAPH_IMPL_HPP
#define CUBBYDNN_GRAPH_IMPL_HPP

namespace CubbyDNN::Core
{
template <typename T>
T* Graph::Node(const std::string& nodeName) const
{
    static_assert(std::is_base_of<Node::Node, T>());

    auto* node = Node(nodeName);

    if (!node)
    {
        return nullptr;
    }

    const auto* nodeType = nodeTypeManager.Type<T>();

    if (!nodeType || !nodeType->IsBaseOf(node->Type()))
    {
        return nullptr;
    }

    return static_cast<T*>(node);
}

template <typename T>
T* Graph::CreateNode(std::string_view nodeName)
{
    static_assert(std::is_base_of<Node::Node, T>());

    const auto* pNodeType(nodeTypeManager.type<T>());

    if (!pNodeType)
    {
        throw std::exception{ "not registered or unknown node type." };
    }

    auto* pNode(Node(nodeName));

    if (!pNode)
        pNode = this->nodeMap
                    .emplace(std::piecewise_construct,
                             std::forward_as_tuple(sNodeName),
                             std::forward_as_tuple(new T(
                                 this, sNodeName, std::forward<P>(tParam)...)))
                    .first->second.get();

    for (const auto *pNodeType{ this->sNodeTypeManager.type<T>() }; pNodeType;
         pNodeType = pNodeType->pBaseType)
        this->sNodeTypeMap.emplace(pNodeType, pNode);

    return static_cast<T *>(pNode);
}
}  // namespace CubbyDNN

#endif