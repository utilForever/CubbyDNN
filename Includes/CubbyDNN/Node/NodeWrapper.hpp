#ifndef CUBBYDNN_NODE_WRAPPER_HPP
#define CUBBYDNN_NODE_WRAPPER_HPP

#include <CubbyDNN/Node/Node.hpp>

namespace CubbyDNN::Node
{
class NodeWrapper
{
 public:
    NodeWrapper(Node* _node);

    Node* const node;
};
}  // namespace CubbyDNN::Node

#endif