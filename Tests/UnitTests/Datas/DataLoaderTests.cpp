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

    std::optional<std::tuple<FloatTensor, LongTensor>> batch;

    for (std::size_t i = 0; i < 5; ++i)
    {
        batch = loader.Next();

        CHECK_EQ(batch.has_value(), true);
        CHECK_EQ(std::get<0>(batch.value()).size(), 4 * 1);
        CHECK_EQ(std::get<0>(batch.value())[0], static_cast<float>(i + 1));
        CHECK_EQ(std::get<1>(batch.value())[0], (i + 1) * 10);
    }

    batch = loader.Next();
    CHECK_EQ(batch.has_value(), false);
}

TEST_CASE("[DataLoader] - Batched(Drop Last)")
{
    DataLoader<TestDataset> loader(std::make_unique<TestDataset>(), 2);

    CHECK_EQ(loader.GetBatchSize(), 2);
    CHECK_EQ(loader.GetSize(), 2);  // 5 = 2 * 2 + 1(dropped)

    loader.Begin();

    std::optional<std::tuple<FloatTensor, LongTensor>> batch;

    long value = 1;
    for (std::size_t i = 0; i < 2; ++i)
    {
        batch = loader.Next();

        CHECK_EQ(batch.has_value(), true);
        CHECK_EQ(std::get<0>(batch.value()).size(), 4 * 2);

        for (int j = 0; j < 2; ++j)
        {
            CHECK_EQ(std::get<0>(batch.value())[j * 4],
                     static_cast<float>(value));

            CHECK_EQ(std::get<1>(batch.value())[j], value * 10);

            ++value;
        }
    }

    batch = loader.Next();
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
    CHECK_EQ(std::get<0>(batch.value()).size(), 4 * 5);

    for (int i = 0; i < 5; ++i)
    {
        CHECK_EQ(std::get<0>(batch.value())[i * 4], static_cast<float>(i + 1));
        CHECK_EQ(std::get<1>(batch.value())[i], (i + 1) * 10);
    }

    batch = loader.Next();
    CHECK_EQ(batch.has_value(), false);
}
