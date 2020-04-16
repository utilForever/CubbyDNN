#ifndef CUBBYDNN_NODE_TYPE_MANAGER_IMPL_HPP
#define CUBBYDNN_NODE_TYPE_MANAGER_IMPL_HPP

namespace CubbyDNN::Node
{
template <class T>
const NodeType* NodeTypeManager::Type() const
{
    static_assert(std::is_base_of<Node, T>());

    return Type(std::string(T::TypeName()));
}

template <typename T>
void NodeTypeManager::RegisterNode()
{
    static_assert(std::is_base_of<Node, T>());

    RegisterNode<T, Node>();
}

template <typename T, typename B>
void NodeTypeManager::RegisterNode()
{
    static_assert(std::is_base_of<B, T>());
    static_assert(std::is_base_of<Node, B>());

    if (Type<T>())
    {
        throw std::exception("already registered node type.");
    }

    const auto* baseType(Type<B>());
    if (!baseType)
    {
        throw std::exception("base type not registered.");
    }

    auto typeName(T::TypeName());
    m_nodeTypeMap.emplace(std::piecewise_construct,
                          std::forward_as_tuple(std::string(typeName)),
                          std::forward_as_tuple(baseType, typeName));
}
}  // namespace CubbyDNN::Node

#endif