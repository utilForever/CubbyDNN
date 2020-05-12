// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "GraphSynchronizationTest.hpp"
#include <cubbydnn/Engine/Model.hpp>
#include "gtest/gtest.h"

namespace GraphTest
{
using namespace CubbyDNN;
using namespace CubbyDNN::Graph;

void GraphExample()
{
    Model graph(NumberSystem::Float);

    const auto placeHolder = graph.PlaceHolder({ 1, 1 });

    const auto dense1 = graph.Dense(placeHolder, 10, Activation::Relu,
                                    std::make_unique<XavierNormal>(),
                                    std::make_unique<LecunNormal>(), 0,
                                    "dense1");

    const auto dense2 =
        graph.Dense(dense1, 5, Activation::Softmax,
                    std::make_unique<XavierNormal>(),
                    std::make_unique<LecunNormal>(), 0.5, "dense2");

    graph.Compile(dense2, OptimizerType::Adam, Loss::CrossEntropy);

    graph.Fit(100);
}

TEST(SimpleGraph, GraphConstruction)
{
    // SimpleGraphTest(25);
}
} // namespace GraphTest
