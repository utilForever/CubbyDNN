// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Engine/Engine.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>
#include <cubbydnn/Units/SinkComputableUnits/SinkUnit.hpp>
#include "gtest/gtest.h"

namespace CubbyDNN
{
void SimpleGraphTestParallel(int numMainThreads, int numCopyThreads,
                             size_t epochs)
{
    /**         hidden1 -- hidden3
     *        /                     \
     * source                         Sink
     *        \ hidden2 -- hidden4  /
     */

    const std::vector<TensorInfo> sourceOutputTensorInfoVector = {
        TensorInfo({ 1, 1, 1, 1 }), TensorInfo({ 1, 1, 1, 1 })
    };

    const std::vector<TensorInfo> inputTensorInfoVector1 = { TensorInfo(
        { 1, 1, 1, 1 }) };
    const std::vector<TensorInfo> outputTensorInfoVector1 = { TensorInfo(
        { 1, 1, 1, 1 }) };

    const std::vector<TensorInfo> inputTensorInfoVector2 = { TensorInfo(
        { 1, 1, 1, 1 }) };
    const std::vector<TensorInfo> outputTensorInfoVector2 = { TensorInfo({
        1,
        1,
        1,
        1,
    }) };

    const std::vector<TensorInfo> inputTensorInfoVector3 = { TensorInfo(
        { 1, 1, 1, 1 }) };
    const std::vector<TensorInfo> outputTensorInfoVector3 = { TensorInfo(
        { 1, 1, 1, 1 }) };

    const std::vector<TensorInfo> inputTensorInfoVector4 = { TensorInfo(
        { 1, 1, 1, 1 }) };
    const std::vector<TensorInfo> outputTensorInfoVector4 = { TensorInfo(
        { 1, 1, 1, 1 }) };

    const std::vector<TensorInfo> sinkInputTensorInfoVector = {
        TensorInfo({ 1, 1, 1, 1 }), TensorInfo({ 1, 1, 1, 1 })
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

    Engine::ConnectSourceToHidden(sourceID, intermediate1ID);
    Engine::ConnectSourceToHidden(sourceID, intermediate3ID);
    Engine::ConnectHiddenToHidden(intermediate1ID, intermediate2ID);
    Engine::ConnectHiddenToHidden(intermediate3ID, intermediate4ID);
    Engine::ConnectHiddenToSink(intermediate2ID, sinkID, 0);
    Engine::ConnectHiddenToSink(intermediate4ID, sinkID, 1);

    Engine::StartExecution(numMainThreads, numCopyThreads, epochs);
    Engine::JoinThreads();
    std::cout << "Terminated" << std::endl;
}

void MultiplyGraphTestParallel(int numMainThreads, int numCopyThreads,
                               size_t epochs)
{
    const TensorInfo inputTensorInfo1 = TensorInfo({ 1, 1, 3, 3 });
    const TensorInfo inputTensorInfo2 = TensorInfo({ 1, 1, 3, 3 });
    const TensorInfo outputTensorInfo = TensorInfo({ 1, 1, 3, 3 });

    void* constantData1 = AllocateData<float>({ 1, 1, 3, 3 });
    void* constantData2 = AllocateData<float>({ 1, 1, 3, 3 });

    SetData<float>({ 0, 0, 0, 0 }, { 1, 1, 3, 3 }, constantData1, 3);
    SetData<float>({ 0, 0, 1, 1 }, { 1, 1, 3, 3 }, constantData1, 3);
    SetData<float>({ 0, 0, 2, 2 }, { 1, 1, 3, 3 }, constantData1, 3);

    SetData<float>({ 0, 0, 0, 0 }, { 1, 1, 3, 3 }, constantData2, 3);
    SetData<float>({ 0, 0, 1, 1 }, { 1, 1, 3, 3 }, constantData2, 3);
    SetData<float>({ 0, 0, 2, 2 }, { 1, 1, 3, 3 }, constantData2, 3);

    const auto testfunction = [](const Tensor& tensor)
    {
        for (size_t rowIdx = 0; rowIdx < 2; ++rowIdx)
            for (size_t colIdx = 0; colIdx < 2; ++colIdx)
            {
                if (rowIdx == colIdx)
                    EXPECT_EQ(GetData<float>({ 0, 0, rowIdx, colIdx }, tensor),
                          9);
                else
                    EXPECT_EQ(GetData<float>({ 0, 0, rowIdx, colIdx }, tensor),
                          0);
            }
    };

    const auto sourceId1 = Engine::Constant(inputTensorInfo1, constantData1);
    const auto sourceId2 = Engine::Constant(inputTensorInfo2, constantData2);

    const auto hiddenId1 =
        Engine::Multiply(inputTensorInfo1, inputTensorInfo2, outputTensorInfo);
    const auto sinkId1 = Engine::AddSinkUnitWithTest(
        { outputTensorInfo }, testfunction);

    Engine::ConnectSourceToHidden(sourceId1, hiddenId1, 0);
    Engine::ConnectSourceToHidden(sourceId2, hiddenId1, 1);
    Engine::ConnectHiddenToSink(hiddenId1, sinkId1);

    Engine::StartExecution(numMainThreads, numCopyThreads, epochs);
    Engine::JoinThreads();
    std::cout << "Terminated MultiplyGraphTestParallel" << std::endl;
}

TEST(SimpleGraphParallel, GraphTestParallel)
{
    SimpleGraphTestParallel(2, 2, 300);
}

TEST(MultiplyGraphParallel, GraphTestParallel)
{
    MultiplyGraphTestParallel(2, 2, 300);
}
} // namespace CubbyDNN
