#ifndef CUBBYDNN_RELU_HPP
#define CUBBYDNN_RELU_HPP

#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Node/Node.hpp>

namespace CubbyDNN::Node
{
class ReLU : public Node
{
 public:
    ReLU(Core::Graph* graph, std::string_view name, float _alpha);
    ReLU(const ReLU& rhs) = delete;
    ReLU(ReLU&& rhs) noexcept = delete;

    virtual ~ReLU() noexcept = default;

    ReLU& operator=(const ReLU& rhs) = delete;
    ReLU& operator=(ReLU&& rhs) noexcept = delete;

    const NodeType* Type() const override;
    static std::string_view TypeName();

    const float alpha = 0.0;

 private:
    NodeInput m_inputLogit;
};
}  // namespace CubbyDNN::Node

#endif