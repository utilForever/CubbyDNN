#include <CubbyDNN/Node/NodeWrapper.hpp>

namespace CubbyDNN::Node
{
NodeWrapper::NodeWrapper(Node* _node) : node(_node)
{
    // Do nothing
}

NodeInput* NodeWrapper::operator[](const std::string& inputName) const
{
    return node->operator[](inputName);
}

NodeWrapper::operator Node*() const noexcept
{
    return node;
}

Core::Span<float> NodeWrapper::Output() const noexcept
{
    return node->Output();
}

NodeWrapper& NodeWrapper::EvalOutput()
{
    node->EvalOutput();

    return *this;
}
}  // namespace CubbyDNN::Node
