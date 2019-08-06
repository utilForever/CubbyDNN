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
    SourceUnit sourceUnit(sourceTensorInfoVector);

    std::vector<TensorInfo> inputTensorInfoVector1 = { TensorInfo(
        { 1, 1, 1 }) };
    std::vector<TensorInfo> outputTensorInfoVector1 = { TensorInfo(
        { 3, 3, 3 }) };
    HiddenUnit intermediateUnit1(inputTensorInfoVector1,
                                 outputTensorInfoVector1);

    std::vector<TensorInfo> inputTensorInfoVector2 = { TensorInfo(
        { 3, 3, 3 }) };
    std::vector<TensorInfo> outputTensorInfoVector2 = { TensorInfo(
        { 6, 6, 6 }) };
    HiddenUnit intermediateUnit2(inputTensorInfoVector2,
                                 outputTensorInfoVector2);

    std::vector<TensorInfo> inputTensorInfoVector3 = { TensorInfo(
        { 1, 2, 1 }) };
    std::vector<TensorInfo> outputTensorInfoVector3 = { TensorInfo(
        { 3, 3, 3 }) };
    HiddenUnit intermediateUnit3(inputTensorInfoVector3,
                                 outputTensorInfoVector3);

    std::vector<TensorInfo> inputTensorInfoVector4 = { TensorInfo(
        { 3, 3, 3 }) };
    std::vector<TensorInfo> outputTensorInfoVector4 = { TensorInfo(
        { 6, 6, 6 }) };
    HiddenUnit intermediateUnit4(inputTensorInfoVector4,
                                 outputTensorInfoVector4);

    std::vector<TensorInfo> sinkTensorInfoVector = { TensorInfo({ 6, 6, 6 }),
                                                     TensorInfo({ 6, 6, 6 }) };
    SinkUnit sinkUnit(sinkTensorInfoVector);

    auto sourceID = Engine::AddSourceUnit(std::move(sourceUnit));
    auto intermediate1ID =
        Engine::AddIntermediateUnit(std::move(intermediateUnit1));
    auto intermediate2ID =
        Engine::AddIntermediateUnit(std::move(intermediateUnit2));
    auto intermediate3ID =
        Engine::AddIntermediateUnit(std::move(intermediateUnit3));
    auto intermediate4ID =
        Engine::AddIntermediateUnit(std::move(intermediateUnit4));
    auto sinkID = Engine::AddSinkUnit(std::move(sinkUnit));

    std::cout<<"A"<<std::endl;
    Engine::ConnectSourceToIntermediate(sourceID, intermediate1ID);
    std::cout<<"B"<<std::endl;
    Engine::ConnectSourceToIntermediate(sourceID, intermediate2ID);
    Engine::ConnectIntermediateToIntermediate(intermediate1ID, intermediate3ID);
    Engine::ConnectIntermediateToIntermediate(intermediate2ID, intermediate4ID);
    Engine::ConnectIntermediateToSink(intermediate3ID, sinkID, 0);
    Engine::ConnectIntermediateToSink(intermediate4ID, sinkID, 1);

    std::cout<<"hi"<<std::endl;
    Engine::StartExecution(2, 2, 100);
    std::cout<<"he"<<std::endl;
    Engine::JoinThreads();
}

void SimpleGraph()
{
    std::vector<TensorInfo> sourceTensorInfoVector = { TensorInfo(
        { 1, 1, 1 }) };
}

TEST(SimpleGraph, GraphConstruction)
{
    GraphConstruction();
}
}  // namespace GraphTest