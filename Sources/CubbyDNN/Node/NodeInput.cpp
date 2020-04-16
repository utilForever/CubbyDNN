#include <CubbyDNN/Node/Node.hpp>
#include <CubbyDNN/Node/NodeInput.hpp>

#include <stdexcept>
#include <utility>

namespace CubbyDNN::Node
{
NodeInput::NodeInput(Node* _node, std::string_view _name,
                     std::function<void(const Node*)> _backwardOp)
    : node(_node),
      name(_name),
      backwardOp(std::move(_backwardOp)),
      m_inputNode(nullptr)
{
    node->m_nodeInputMap[name] = this;
}

NodeInput::operator bool() const
{
    return static_cast<bool>(m_inputNode);
}

Node* NodeInput::InputNode() noexcept
{
    return m_inputNode;
}

const Node* NodeInput::InputNode() const noexcept
{
    return m_inputNode;
}

bool NodeInput::IsDependOn(const Node* node) const
{
    return m_depsSet.count(const_cast<Node*>(node));
}

void NodeInput::Attach(Node* inputNode)
{
    if (!inputNode)
    {
        throw std::runtime_error("unable to attach null node");
    }

    if (m_inputNode)
    {
        return;
    }

    // Trace dependencies and merge into the owner node.
    m_depsSet.emplace(inputNode);
    m_depsSet.insert(inputNode->m_deps.cbegin(), inputNode->m_deps.cend());
    node->m_deps.insert(m_depsSet.cbegin(), m_depsSet.cend());

    // Trace reverse dependencies.
    inputNode->m_revNodeInputList.emplace_back(this);

    for (auto* dep : m_depsSet)
    {
        dep->m_revDeps.emplace(node);
        dep->m_revDeps.insert(node->m_revDeps.cbegin(), node->m_revDeps.cend());
    }

    m_inputNode = inputNode;
}
}  // namespace CubbyDNN::Node