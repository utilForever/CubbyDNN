#include <doctest.h>
#include <iostream>

#include <CubbyDNN/Datas/Dataset.hpp>
#include <CubbyDNN/Datas/TransformedDataset.hpp>
#include <CubbyDNN/Preprocess/Transforms.hpp>

#include "TestDataset.hpp"

using namespace CubbyDNN;

TEST_CASE("[TransformedDataset] - Single transform")
{
    auto dataset = TestDataset().Transform(
        Transforms::Normalize({ 3 }, { 2 }, { 1, 1, 1 }));

    CHECK_EQ(dataset.GetSize(), 5);

    const auto data = dataset.Get(0);
    CHECK_EQ(data.Data.GetSpan()[0], doctest::Approx(-1));
}

TEST_CASE("[TransformedDataset] - Multiple transforms")
{
    auto dataset =
        TestDataset()
            .Transform(Transforms::Normalize({ 3 }, { 2 }, { 1, 1, 1 }))
            .Transform(Transforms::Normalize({ 1 }, { 5 }, { 1, 1, 1 }));
    
    CHECK_EQ(dataset.GetSize(), 5);

    const auto data = dataset.Get(3);
    CHECK_EQ(data.Data.GetSpan()[0], doctest::Approx(-0.1));
}
