#ifndef CUBBYDNN_NODE_INPUT_HPP
#define CUBBYDNN_NODE_INPUT_HPP

#include <functional>
#include <string>
#include <unordered_set>

namespace CubbyDNN::Node
{
class Node;

class NodeInput
{
 public:
    NodeInput(Node* _node, std::string_view _name,
              std::function<void(const Node*)> _backwardOp);

    operator bool() const;

    Node* InputNode() noexcept;
    const Node* InputNode() const noexcept;

    bool IsDependOn(const Node* _node) const;
    void Attach(Node* inputNode);

    Node* const node;
    const std::string name;
    const std::function<void(const Node*)> backwardOp;

 private:
    Node* m_inputNode;
    std::unordered_set<Node*> m_depsSet;
};
}  // namespace CubbyDNN::Node

#endif