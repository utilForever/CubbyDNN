#include <CubbyDNN/Node/Input.hpp>

namespace CubbyDNN::Node
{
Input::Input(Core::Graph* graph, std::string_view name) : Node(graph, name)
{
    // Do nothing
}

const NodeType* Input::Type() const
{
    return graph->nodeTypeManager.Type<Input>();
}

std::string_view Input::TypeName()
{
    return "Input";
}

void Input::Feed(const Core::Shape& shape, Core::Span<float> span)
{
    m_inputSpan = span;

    const bool isDirtyShape = m_inputShape != shape;
    if (isDirtyShape)
    {
        m_inputShape = shape;
    }

    MarkDirty(isDirtyShape);
}
}  // namespace CubbyDNN::Node