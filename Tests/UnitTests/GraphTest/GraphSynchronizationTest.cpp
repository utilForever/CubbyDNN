// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "GraphSynchronizationTest.hpp"
#include <cubbydnn/Engine/Engine.hpp>
#include "gtest/gtest.h"

namespace GraphTest
{
using namespace CubbyDNN;

void GraphExample()
{
    Graph graph(NumberSystem::Float);
    const auto placeHolder = graph.PlaceHolder({ 1, 1 });
    const auto dense1 = graph.Dense(placeHolder, 10, Activation::Relu, 
        InitializerType::HeNormal,
                InitializerType::HeNormal);
    const auto dense2 =
        graph.Dense(dense1, 5, Activation::Softmax,
                    InitializerType::LeCunNormal, InitializerType::LeCunNormal);
    graph.Compile(OptimizerType::Adam, Loss::CrossEntropy);

    graph.Fit(100);

}

TEST(SimpleGraph, GraphConstruction)
{
    // SimpleGraphTest(25);
}

}  // namespace GraphTest
