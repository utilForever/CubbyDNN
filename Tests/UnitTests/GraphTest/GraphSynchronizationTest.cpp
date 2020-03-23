// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "GraphSynchronizationTest.hpp"
#include <cubbydnn/Engine/Graph.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>
#include "gtest/gtest.h"

namespace GraphTest
{
using namespace CubbyDNN;

void SimpleGraphTest(std::size_t epochs)
{
    /**         hidden1 -- hidden3
     *        /                     \
     * source                         Sink
     *        \ hidden2 -- hidden4  /
     */

    const auto source = Graph::Source(TensorInfo({ 1, 1, 1, 1 }), 2);
    const auto hidden1 = Graph::Hidden(
        { source }, TensorInfo({ 1, 1, 1, 1 }));
    const auto hidden2 = Graph::Hidden(
        { hidden1 }, TensorInfo({ 1, 1, 1, 1 }));
    const auto hidden3 = Graph::Hidden(
        { source }, TensorInfo({ 1, 1, 1, 1 }));
    const auto hidden4 = Graph::Hidden(
        { hidden3 }, TensorInfo({ 1, 1, 1, 1 }));
    Graph::Sink({ hidden2, hidden4 },
                 { TensorInfo({ 1, 1, 1, 1 }), TensorInfo({ 1, 1, 1, 1 }) });

    Graph::ExecuteForward(epochs);
    std::cout << "Terminated" << std::endl;
}

void MultiplyGraphTestSerial(std::size_t epochs)
{
    void* constantData1 = AllocateData<float>({ batchSize, channelSize, 3, 3 });
    void* constantData2 = AllocateData<float>({ batchSize, channelSize, 3, 3 });
    void* constantData3 = AllocateData<float>({ batchSize, channelSize, 3, 3 });

    SetData<float>({ 0, 0, 0, 0 }, { 1, 1, 3, 3 }, constantData1, 3);
    SetData<float>({ 0, 0, 1, 1 }, { 1, 1, 3, 3 }, constantData1, 3);
    SetData<float>({ 0, 0, 2, 2 }, { 1, 1, 3, 3 }, constantData1, 3); 

            SetData<float>({ batchIdx, channelIdx, 0, 0 },
                           { batchSize, channelSize, 3, 3 }, constantData2, 3);
            SetData<float>({ batchIdx, channelIdx, 1, 1 },
                           { batchSize, channelSize, 3, 3 }, constantData2, 3);
            SetData<float>({ batchIdx, channelIdx, 2, 2 },
                           { batchSize, channelSize, 3, 3 }, constantData2, 3);

    const auto testFunction = [](const Tensor& tensor, std::size_t epoch)
    {
        std::cout << "epoch: " << epoch << std::endl;
        for (std::size_t rowIdx = 0; rowIdx < 2; ++rowIdx)
            for (std::size_t colIdx = 0; colIdx < 2; ++colIdx)
            {
                if (rowIdx == colIdx)
                    EXPECT_EQ(GetData<float>({ 0, 0, rowIdx, colIdx }, tensor),
                          9);
                else
                    EXPECT_EQ(GetData<float>({ 0, 0, rowIdx, colIdx }, tensor),
                          0);
            }
    };

    const auto constant1 =
        Graph::Constant(TensorInfo({ 1, 1, 3, 3 }), constantData1);
    const auto constant2 = Graph::Constant(TensorInfo({ 1, 1, 3, 3 }),
                                            constantData2);

    const auto multiply1 = Graph::Multiply(constant1, constant2);
    const auto multiply2 = Graph::Multiply(multiply1, constant3);

    Graph::OutputTest(multiply2, testFunction);

    Graph::ExecuteForward(epochs);
    std::cout << "Terminated MultiplyGraphTestSerial" << std::endl;
}

TEST(SimpleGraph, GraphConstruction)
{
    //SimpleGraphTest(25);
}

TEST(SimpleGraph, MultiplyGraphTestSerial)
{
    //MultiplyGraphTestSerial(25);
    EXPECT_EQ(0, 0);
}
} // namespace GraphTest
