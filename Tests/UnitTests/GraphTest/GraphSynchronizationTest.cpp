// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "GraphSynchronizationTest.hpp"

namespace GraphTest
{
using namespace CubbyDNN;

void SimpleGraphTest()
{
    /**         hidden1 -- hidden3
     *        /                     \
     * source                         Sink
     *        \ hidden2 -- hidden4  /
     */

    const std::vector<TensorInfo> sourceOutputTensorInfoVector = {
        TensorInfo({}), TensorInfo({})
    };

    const std::vector<TensorInfo> inputTensorInfoVector1 = { TensorInfo({}) };
    const std::vector<TensorInfo> outputTensorInfoVector1 = { TensorInfo({}) };

    const std::vector<TensorInfo> inputTensorInfoVector2 = { TensorInfo({}) };
    const std::vector<TensorInfo> outputTensorInfoVector2 = { TensorInfo({}) };

    const std::vector<TensorInfo> inputTensorInfoVector3 = { TensorInfo({}) };
    const std::vector<TensorInfo> outputTensorInfoVector3 = { TensorInfo({}) };

    const std::vector<TensorInfo> inputTensorInfoVector4 = { TensorInfo({}) };
    const std::vector<TensorInfo> outputTensorInfoVector4 = { TensorInfo(
        {}) };

    const std::vector<TensorInfo> sinkInputTensorInfoVector = {
        TensorInfo({}),
        TensorInfo({})
    };

    SinkUnit sinkUnit = SinkUnit(sinkInputTensorInfoVector);

    const auto sourceID = Engine::AddSourceUnit(sourceOutputTensorInfoVector);
    const auto intermediate1ID =
        Engine::AddHiddenUnit(inputTensorInfoVector1, inputTensorInfoVector2);
    const auto intermediate2ID =
        Engine::AddHiddenUnit(inputTensorInfoVector2, outputTensorInfoVector2);
    const auto intermediate3ID =
        Engine::AddHiddenUnit(inputTensorInfoVector3, outputTensorInfoVector3);
    const auto intermediate4ID =
        Engine::AddHiddenUnit(inputTensorInfoVector4, outputTensorInfoVector4);
    const auto sinkID = Engine::AddSinkUnit(sinkInputTensorInfoVector);

    Engine::ConnectSourceToIntermediate(sourceID, intermediate1ID);
    Engine::ConnectSourceToIntermediate(sourceID, intermediate3ID);
    Engine::ConnectIntermediateToIntermediate(intermediate1ID, intermediate2ID);
    Engine::ConnectIntermediateToIntermediate(intermediate3ID, intermediate4ID);
    Engine::ConnectIntermediateToSink(intermediate2ID, sinkID, 0);
    Engine::ConnectIntermediateToSink(intermediate4ID, sinkID, 1);

    Engine::StartExecution(100);
    std::cout << "Terminated" << std::endl;
}

void SimpleGraph()
{
    std::vector<TensorInfo> sourceTensorInfoVector = { TensorInfo(
        {}) };
}

TEST(SimpleGraph, GraphConstruction)
{
    SimpleGraphTest();
    EXPECT_EQ(0, 0);
}
} // namespace GraphTest
