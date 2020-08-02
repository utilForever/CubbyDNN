#ifndef CUBBYDNN_GRAPH_IMPL_HPP
#define CUBBYDNN_GRAPH_IMPL_HPP

#include <stdexcept>

namespace CubbyDNN::Core
{
template <typename T>
std::size_t Graph::NodeCount() const
{
    return NodeCount(nodeTypeManager.Type<T>());
}

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

template <typename T, typename... P>
T* Graph::CreateNode(const std::string& nodeName, P&&... params)
{
    static_assert(std::is_base_of<Node::Node, T>());

    const auto* pNodeType(nodeTypeManager.Type<T>());

    if (!pNodeType)
    {
        throw std::invalid_argument("not registered or unknown node type.");
    }

    auto node = Node(nodeName);

    if (!node)
    {
        node = m_nodeMap
                   .emplace(std::piecewise_construct,
                            std::forward_as_tuple(nodeName),
                            std::forward_as_tuple(new T(
                                this, nodeName, std::forward<P>(params)...)))
                   .first->second.get();
    }

    for (const auto* nodeType = nodeTypeManager.Type<T>(); nodeType;
         nodeType = nodeType->baseType)
    {
        m_nodeTypeMap.emplace(nodeType, node);
    }

    return static_cast<T*>(node);
}

template <typename T, typename... P>
T* Graph::CreateInitializer(P&&... params)
{
    static_assert(std::is_base_of<Initializer::Initializer, T>());

    return static_cast<T*>(
        this->m_intializerSet.emplace(new T(std::forward<P>(params)...))
            .first->get());
}
}  // namespace CubbyDNN::Core

#endif