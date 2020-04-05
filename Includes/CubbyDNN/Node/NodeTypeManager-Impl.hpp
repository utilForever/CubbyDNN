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
}  // namespace CubbyDNN::Node

#endif