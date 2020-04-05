#ifndef CUBBYDNN_GRAPH_BUILDER_HPP
#define CUBBYDNN_GRAPH_BUILDER_HPP

#include <CubbyDNN/Node/NodeWrapper.hpp>

#include <string_view>

namespace CubbyDNN::Core
{
class Graph;

class GraphBuilder
{
 public:
    Node::NodeWrapper Input(std::string_view nodeName);

    const Graph* graph;
};
}  // namespace CubbyDNN::Core

#endif