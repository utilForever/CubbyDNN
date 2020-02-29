// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Engine/Engine.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>
#include "gtest/gtest.h"

namespace CubbyDNN
{
void SimpleGraphTestParallel(int workers,
                             size_t epochs)
{
    /**         hidden1 -- hidden3
     *        /                     \
     * source                         Sink
     *        \ hidden2 -- hidden4  /
     */

    const auto source = Engine::Source(TensorInfo({ 1, 1, 1, 1 }), 2);
    const auto hidden1 =
        Engine::Hidden({ source }, TensorInfo({ 1, 1, 1, 1 }),
                       1);
    const auto hidden2 =
        Engine::Hidden({ hidden1 }, TensorInfo({ 1, 1, 1, 1 }));
    const auto hidden3 = Engine::Hidden(
        { source }, TensorInfo({ 1, 1, 1, 1 }));
    const auto hidden4 =
        Engine::Hidden({ hidden3 }, TensorInfo({ 1, 1, 1, 1 }));
    Engine::Sink({ hidden2, hidden4 },
                 { TensorInfo({ 1, 1, 1, 1 }), TensorInfo({ 1, 1, 1, 1 }) });

    Engine::ExecuteParallel(workers, epochs);
    Engine::JoinThreads();
    std::cout << "Terminated" << std::endl;
}

void MultiplyGraphTestParallel(size_t workers, size_t epochs)
{

    void* constantData1 = AllocateData<float>({ 1, 1, 3, 3 });
    void* constantData2 = AllocateData<float>({ 1, 1, 3, 3 });
    void* constantData3 = AllocateData<float>({ 1, 1, 3, 3 });


    SetData<float>({ 0, 0, 0, 0 }, { 1, 1, 3, 3 }, constantData1, 3);
    SetData<float>({ 0, 0, 1, 1 }, { 1, 1, 3, 3 }, constantData1, 3);
    SetData<float>({ 0, 0, 2, 2 }, { 1, 1, 3, 3 }, constantData1, 3);

    SetData<float>({ 0, 0, 0, 0 }, { 1, 1, 3, 3 }, constantData2, 3);
    SetData<float>({ 0, 0, 1, 1 }, { 1, 1, 3, 3 }, constantData2, 3);
    SetData<float>({ 0, 0, 2, 2 }, { 1, 1, 3, 3 }, constantData2, 3);

    SetData<float>({ 0, 0, 0, 0 }, { 1, 1, 3, 3 }, constantData3, 3);
    SetData<float>({ 0, 0, 1, 1 }, { 1, 1, 3, 3 }, constantData3, 3);
    SetData<float>({ 0, 0, 2, 2 }, { 1, 1, 3, 3 }, constantData3, 3);

    const auto testFunction = [](const Tensor& tensor)
    {
        for (size_t rowIdx = 0; rowIdx < 2; ++rowIdx)
            for (size_t colIdx = 0; colIdx < 2; ++colIdx)
            {
                if (rowIdx == colIdx)
                    EXPECT_EQ(GetData<float>({ 0, 0, rowIdx, colIdx }, tensor),
                          27);
                else
                    EXPECT_EQ(GetData<float>({ 0, 0, rowIdx, colIdx }, tensor),
                          0);
            }
    };

    const auto constant1 = Engine::Constant(TensorInfo({ 1, 1, 3, 3 }),
                                            constantData1);
    const auto constant2 = Engine::Constant(TensorInfo({ 1, 1, 3, 3, }),
                                            constantData2);
    const auto constant3 = Engine::Constant(TensorInfo({
                                                1,
                                                1,
                                                3,
                                                3,
                                            }),
                                            constantData3);

    const auto multiply1 = Engine::Multiply(constant1, constant2);
    const auto multiply2 = Engine::Multiply(multiply1, constant3);

    Engine::OutputTest(multiply2, testFunction);

    Engine::ExecuteParallel(workers, epochs);
    Engine::JoinThreads();
    std::cout << "Terminated MultiplyGraphTestParallel" << std::endl;
}

TEST(SimpleGraphParallel, GraphTestParallel)
{
    SimpleGraphTestParallel(2, 300);
}

TEST(MultiplyGraphParallel, GraphTestParallel)
{
    MultiplyGraphTestParallel(2, 10000);
}
} // namespace CubbyDNN
