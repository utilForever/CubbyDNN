//
// Created by jwkim98 on 4/19/19.
//

#include "GraphSynchronizationTest.hpp"

namespace GraphTest
{
using namespace CubbyDNN;
void GraphConstruction()
{
    /**         hidden1 -- hidden3
     *        /                     \
     * source                         Sink
     *        \ hidden2 -- hidden4  /
     */

    std::vector<TensorInfo> sourceTensorInfoVector = {
        TensorInfo({ 1, 1, 1 }), TensorInfo({ 1, 2, 1 })
    };
    SourceUnit* sourceUnit = new SourceUnit(sourceTensorInfoVector);

    std::vector<TensorInfo> inputTensorInfoVector1 = { TensorInfo(
        { 1, 1, 1 }) };
    std::vector<TensorInfo> outputTensorInfoVector1 = { TensorInfo(
        { 3, 3, 3 }) };
    HiddenUnit* hiddenUnit1 =
        new HiddenUnit(inputTensorInfoVector1, outputTensorInfoVector1);

    std::vector<TensorInfo> inputTensorInfoVector2 = { TensorInfo(
        { 3, 3, 3 }) };
    std::vector<TensorInfo> outputTensorInfoVector2 = { TensorInfo(
        { 6, 6, 6 }) };
    HiddenUnit* hiddenUnit2 =
        new HiddenUnit(inputTensorInfoVector2, outputTensorInfoVector2);

    std::vector<TensorInfo> inputTensorInfoVector3 = { TensorInfo(
        { 1, 2, 1 }) };
    std::vector<TensorInfo> outputTensorInfoVector3 = { TensorInfo(
        { 3, 3, 3 }) };
    HiddenUnit* hiddenUnit3 =
        new HiddenUnit(inputTensorInfoVector3, outputTensorInfoVector3);

    std::vector<TensorInfo> inputTensorInfoVector4 = { TensorInfo(
        { 3, 3, 3 }) };
    std::vector<TensorInfo> outputTensorInfoVector4 = { TensorInfo(
        { 6, 6, 6 }) };
    HiddenUnit* hiddenUnit4 =
        new HiddenUnit(inputTensorInfoVector4, outputTensorInfoVector4);

    std::vector<TensorInfo> sinkTensorInfoVector = { TensorInfo({ 6, 6, 6 }),
                                                     TensorInfo({ 6, 6, 6 }) };
    SinkUnit* sinkUnit = new SinkUnit(sinkTensorInfoVector);

    auto sourceID = Engine::AddSourceUnit(std::move(sourceUnit));
    auto intermediate1ID = Engine::AddHiddenUnit(std::move(hiddenUnit1));
    auto intermediate2ID = Engine::AddHiddenUnit(std::move(hiddenUnit2));
    auto intermediate3ID = Engine::AddHiddenUnit(std::move(hiddenUnit3));
    auto intermediate4ID = Engine::AddHiddenUnit(std::move(hiddenUnit4));
    auto sinkID = Engine::AddSinkUnit(std::move(sinkUnit));

    Engine::ConnectSourceToIntermediate(sourceID, intermediate1ID);
    Engine::ConnectSourceToIntermediate(sourceID, intermediate3ID);
    Engine::ConnectIntermediateToIntermediate(intermediate1ID, intermediate2ID);
    Engine::ConnectIntermediateToIntermediate(intermediate3ID, intermediate4ID);
    Engine::ConnectIntermediateToSink(intermediate2ID, sinkID, 0);
    Engine::ConnectIntermediateToSink(intermediate4ID, sinkID, 1);

    Engine::StartExecution(3, 3, 100);
    Engine::JoinThreads();
    Engine::ReleaseResources();
    std::cout<<"Terminated"<<std::endl;
}

void SimpleGraph()
{
    std::vector<TensorInfo> sourceTensorInfoVector = { TensorInfo(
        { 1, 1, 1 }) };
}

TEST(SimpleGraph, GraphConstruction)
{
    GraphConstruction();
    EXPECT_EQ(0,0);
}
}  // namespace GraphTest