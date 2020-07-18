#include <doctest.h>
#include <iostream>

#include <CubbyDNN/Datas/Dataset.hpp>
#include <CubbyDNN/Datas/TransformedDataset.hpp>
#include <CubbyDNN/Preprocess/Transforms.hpp>

#include "TestDataset.hpp"

using namespace CubbyDNN;

TEST_CASE("[TransformedDataset] - Single transform")
{
    auto dataset = TransformedDataset(
        TestDataset(), Transforms::Normalize<long>({ 3 }, { 2 }));

    CHECK_EQ(dataset.GetSize(), 5);

    const auto data = dataset.Get(0);
    CHECK_EQ(data.Data[0], doctest::Approx(-1));
    CHECK_EQ(data.Target, 10);
}

TEST_CASE("[TransformedDataset] - Multiple transforms")
{
    auto dataset = TransformedDataset(
        TransformedDataset(TestDataset(),
                           Transforms::Normalize<long>({ 3 }, { 2 })),
        Transforms::Normalize<long>({ 1 }, { 5 }));

    CHECK_EQ(dataset.GetSize(), 5);

    const auto data = dataset.Get(3);
    CHECK_EQ(data.Data[0], doctest::Approx(-0.1));
    CHECK_EQ(data.Target, 40);
}
