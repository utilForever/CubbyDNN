#ifndef CUBBYDNN_SOFTMAX_HPP
#define CUBBYDNN_SOFTMAX_HPP

#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Node/Node.hpp>

namespace CubbyDNN::Node
{
class Softmax final : public Node
{
 public:
    Softmax(Core::Graph* graph, std::string_view name,
            const std::vector<bool>& _groupAxis);
    Softmax(const Softmax& rhs) = delete;
    Softmax(Softmax&& rhs) noexcept = delete;

    virtual ~Softmax() noexcept = default;

    Softmax& operator=(const Softmax& rhs) = delete;
    Softmax& operator=(Softmax&& rhs) noexcept = delete;

    const NodeType* Type() const override;
    static std::string_view TypeName();

    const std::vector<bool> groupAxis;

 private:
    void EvalShapeInternal() override;
    void EvalOutputInternal() override;

    void BackwardOp(const Node* dy);

    Core::Memory<float> m_summation;
    NodeInput m_inputLogit;
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>>
        m_indexFactorList;
};
}  // namespace CubbyDNN::Node

#endif