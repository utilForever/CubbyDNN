#ifndef CUBBYDNN_MOMENTUM_HPP
#define CUBBYDNN_MOMENTUM_HPP

#include <CubbyDNN/Core/Memory.hpp>
#include <CubbyDNN/Node/Parameter.hpp>

namespace CubbyDNN::Optimizer
{
class Momentum
{
 public:
    Momentum(float _momentum,
             std::initializer_list<Node::Parameter*> parameterList);
    Momentum(float _momentum, std::vector<Node::Parameter*> parameterList);
    Momentum(const Momentum& rhs) = default;
    Momentum(Momentum&& rhs) noexcept = default;
    ~Momentum() noexcept = default;

    Momentum& operator=(const Momentum& rhs) = delete;
    Momentum& operator=(Momentum&& rhs) noexcept = delete;

    void Reduce(float learningRate, Node::Node* target);

    const float momentum = 0.0f;

 private:
    std::vector<Node::Parameter*> m_parameterList;
    std::vector<Core::Memory<float>> m_momentumGradientList;
};
}  // namespace CubbyDNN::Optimizer

#endif