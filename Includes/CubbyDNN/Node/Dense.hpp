#ifndef CUBBYDNN_DENSE_HPP
#define CUBBYDNN_DENSE_HPP

#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Node/Node.hpp>

namespace CubbyDNN::Node
{
class Dense final : public Node
{
 public:
    Dense(Core::Graph* graph, std::string_view name);
    Dense(const Dense& rhs) = delete;
    Dense(Dense&& rhs) noexcept = delete;

    virtual ~Dense() noexcept = default;

    Dense& operator=(const Dense& rhs) = delete;
    Dense& operator=(Dense&& rhs) noexcept = delete;

    const NodeType* Type() const override;
    static std::string_view TypeName();

 private:
    void EvalShapeInternal() override;
    void EvalOutputInternal() override;

    void BackwardOpInput(const Node* dy);
    void BackwardOpWeight(const Node* dy);
    void BackwardOpBias(const Node* dy);

    NodeInput m_input;
    NodeInput m_inputWeight;
    NodeInput m_inputBias;
};
}  // namespace CubbyDNN::Node

#endif