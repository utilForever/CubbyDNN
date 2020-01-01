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

    const std::vector<TensorInfo> sourceTensorInfoVector = {
        TensorInfo({ 1, 1, 1 }), TensorInfo({ 1, 2, 1 })
    };
    SourceUnit sourceUnit = SourceUnit(sourceTensorInfoVector);

    const std::vector<TensorInfo> inputTensorInfoVector1 = { TensorInfo(
        { 1, 1, 1 }) };
    const std::vector<TensorInfo> outputTensorInfoVector1 = { TensorInfo(
        { 3, 3, 3 }) };
    HiddenUnit hiddenUnit1 =
        HiddenUnit(inputTensorInfoVector1, outputTensorInfoVector1);

    const std::vector<TensorInfo> inputTensorInfoVector2 = { TensorInfo(
        { 3, 3, 3 }) };
    const std::vector<TensorInfo> outputTensorInfoVector2 = { TensorInfo(
        { 6, 6, 6 }) };
    HiddenUnit hiddenUnit2 =
        HiddenUnit(inputTensorInfoVector2, outputTensorInfoVector2);

    const std::vector<TensorInfo> inputTensorInfoVector3 = { TensorInfo(
        { 1, 2, 1 }) };
    const std::vector<TensorInfo> outputTensorInfoVector3 = { TensorInfo(
        { 3, 3, 3 }) };
    HiddenUnit hiddenUnit3 =
        HiddenUnit(inputTensorInfoVector3, outputTensorInfoVector3);

    const std::vector<TensorInfo> inputTensorInfoVector4 = { TensorInfo(
        { 3, 3, 3 }) };
    const std::vector<TensorInfo> outputTensorInfoVector4 = { TensorInfo(
        { 6, 6, 6 }) };
    HiddenUnit hiddenUnit4 =
        HiddenUnit(inputTensorInfoVector4, outputTensorInfoVector4);

    const std::vector<TensorInfo> sinkTensorInfoVector = { TensorInfo({ 6, 6, 6 }),
                                                     TensorInfo({ 6, 6, 6 }) };
    SinkUnit sinkUnit = SinkUnit(sinkTensorInfoVector);

    const auto sourceID = Engine::AddSourceUnit(std::move(sourceUnit));
    const auto intermediate1ID = Engine::AddHiddenUnit(std::move(hiddenUnit1));
    const auto intermediate2ID = Engine::AddHiddenUnit(std::move(hiddenUnit2));
    const auto intermediate3ID = Engine::AddHiddenUnit(std::move(hiddenUnit3));
    const auto intermediate4ID = Engine::AddHiddenUnit(std::move(hiddenUnit4));
    const auto sinkID = Engine::AddSinkUnit(std::move(sinkUnit));

    Engine::ConnectSourceToIntermediate(sourceID, intermediate1ID);
    Engine::ConnectSourceToIntermediate(sourceID, intermediate3ID);
    Engine::ConnectIntermediateToIntermediate(intermediate1ID, intermediate2ID);
    Engine::ConnectIntermediateToIntermediate(intermediate3ID, intermediate4ID);
    Engine::ConnectIntermediateToSink(intermediate2ID, sinkID, 0);
    Engine::ConnectIntermediateToSink(intermediate4ID, sinkID, 1);

    Engine::StartExecution(3, 3, 100);
    Engine::JoinThreads();
    std::cout<<"Terminated"<<std::endl;
}

void SimpleGraph()
{
    std::vector<TensorInfo> sourceTensorInfoVector = { TensorInfo(
        { 1, 1, 1 }) };
}

TEST(SimpleGraph, GraphConstruction)
{
    SimpleGraphTest();
    EXPECT_EQ(0,0);
}
}  // namespace GraphTest