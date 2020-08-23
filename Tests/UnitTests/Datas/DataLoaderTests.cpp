#include <doctest.h>
#include <iostream>

#include <CubbyDNN/Datas/DataLoader.hpp>
#include <CubbyDNN/Datas/Dataset.hpp>

#include "TestDataset.hpp"

using namespace CubbyDNN;

TEST_CASE("[DataLoader] - Big batchsize than dataset size")
{
    CHECK_THROWS(
        DataLoader<TestDataset> loader(std::make_unique<TestDataset>(), 500));
}

TEST_CASE("[DataLoader] - NotBatched")
{
    DataLoader<TestDataset> loader(std::make_unique<TestDataset>());

    CHECK_EQ(loader.GetBatchSize(), 1);
    CHECK_EQ(loader.GetSize(), 5);

    loader.Begin();

    for (std::size_t i = 0; i < 5; ++i)
    {
        auto batch = loader.Next();

        CHECK_EQ(batch.has_value(), true);
        CHECK_EQ(batch.value().Data.GetSpan().Length(), 4 * 1);
        CHECK_EQ(batch.value().Data.GetSpan()[0], static_cast<float>(i + 1));
    }

    auto batch = loader.Next();
    CHECK_EQ(batch.has_value(), false);
}

TEST_CASE("[DataLoader] - Batched(Drop Last)")
{
    DataLoader<TestDataset> loader(std::make_unique<TestDataset>(), 2);

    CHECK_EQ(loader.GetBatchSize(), 2);
    CHECK_EQ(loader.GetSize(), 2);  // 5 = 2 * 2 + 1(dropped)

    loader.Begin();

    long value = 1;
    for (std::size_t i = 0; i < 2; ++i)
    {
        auto batch = loader.Next();

        CHECK_EQ(batch.has_value(), true);

        for (int j = 0; j < 2; ++j)
        {
            CHECK_EQ(batch.value().Data.GetSpan()[j * 4],
                     static_cast<float>(value));

            ++value;
        }
    }

    auto batch = loader.Next();
    CHECK_EQ(batch.has_value(), false);
}

TEST_CASE("[DataLoader] - Batched(Not Drop Last)")
{
    DataLoader<TestDataset> loader(std::make_unique<TestDataset>(), 5);

    CHECK_EQ(loader.GetBatchSize(), 5);
    CHECK_EQ(loader.GetSize(), 1);  // 5 = 5 * 1

    loader.Begin();

    auto batch = loader.Next();

    CHECK_EQ(batch.has_value(), true);
    CHECK_EQ(batch.value().Data.GetSpan().Length(), 4 * 5);

    for (int i = 0; i < 5; ++i)
    {
        CHECK_EQ(batch.value().Data.GetSpan()[i * 4], static_cast<float>(i + 1));
    }

    batch = loader.Next();
    CHECK_EQ(batch.has_value(), false);
}
