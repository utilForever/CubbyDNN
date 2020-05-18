#ifndef CUBBYDNN_INPUT_HPP
#define CUBBYDNN_INPUT_HPP

#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Node/Node.hpp>

namespace CubbyDNN::Node
{
class Input : public Node
{
 public:
    Input(Core::Graph* graph, std::string_view name);
    Input(const Input& rhs) = delete;
    Input(Input&& rhs) noexcept = delete;

    virtual ~Input() noexcept = default;

    Input& operator=(const Input& rhs) = delete;
    Input& operator=(Input&& rhs) noexcept = delete;

    const NodeType* Type() const override;
    static std::string_view TypeName();

    void Feed(const Core::Shape& shape, Core::Span<float> span);

 private:
    Core::Shape m_inputShape;
    Core::Span<float> m_inputSpan;
};
}  // namespace CubbyDNN::Node

#endif