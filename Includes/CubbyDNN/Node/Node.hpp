#ifndef CUBBYDNN_NODE_HPP
#define CUBBYDNN_NODE_HPP

#include <CubbyDNN/Core/Memory.hpp>
#include <CubbyDNN/Core/Shape.hpp>
#include <CubbyDNN/Node/NodeInput.hpp>
#include <CubbyDNN/Node/NodeType.hpp>

#include <unordered_set>

namespace CubbyDNN::Core
{
class Graph;
}

namespace CubbyDNN::Node
{
class Node
{
 public:
    friend NodeInput;

    Node(Core::Graph* _graph, std::string_view _name);

    NodeInput* operator[](const std::string& inputName);

    virtual const NodeType* Type() const;
    static std::string_view TypeName();

    const Core::Shape& Shape() const noexcept;
    Core::Span<float> Output() const noexcept;
    Core::Span<float> Gradient() const noexcept;

    bool HasRevDeps(const Node* revDep) const;

    Node& MarkDirty(bool dirtyShape = true);

    Node& EvalShape();
    Node& EvalOutput();
    Node& EvalGradient(const Node* dy);

    Core::Graph* const graph;
    const std::string name;

 protected:
    Core::Shape m_shape;
    Core::Memory<float> m_output;
    Core::Memory<float> m_gradient;
    std::vector<NodeInput*> m_revNodeInputList;
    std::unordered_set<Node*> m_deps;
    std::unordered_set<Node*> m_revDeps;
    std::unordered_map<std::string, NodeInput*> m_nodeInputMap;

 private:
    bool m_isShapeDirty;
    bool m_isOutputDirty;
    const Node* m_gradientDirty;
};
}  // namespace CubbyDNN::Node

#endif