#ifndef CUBBYDNN_SOFTMAX_CE_HPP
#define CUBBYDNN_SOFTMAX_CE_HPP

#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Node/Node.hpp>

namespace CubbyDNN::Node
{
class SoftmaxCE final : public Node
{
 public:
    SoftmaxCE(Core::Graph* graph, std::string_view name);
    SoftmaxCE(const SoftmaxCE& rhs) = delete;
    SoftmaxCE(SoftmaxCE&& rhs) noexcept = delete;

    virtual ~SoftmaxCE() noexcept = default;

    SoftmaxCE& operator=(const SoftmaxCE& rhs) = delete;
    SoftmaxCE& operator=(SoftmaxCE&& rhs) noexcept = delete;

    const NodeType* Type() const override;
    static std::string_view TypeName();

 private:
    void EvalShapeInternal() override;
    void EvalOutputInternal() override;

    void BackwardOpLabel(const Node* dy);
    void BackwardOpProb(const Node* dy);

    NodeInput m_inputLabel;
    NodeInput m_inputProb;
};
}  // namespace CubbyDNN::Node

#endif