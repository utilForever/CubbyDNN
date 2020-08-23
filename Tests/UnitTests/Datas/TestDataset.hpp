#ifndef CUBBYDNN_TEST_TEST_DATASET_HPP
#define CUBBYDNN_TEST_TEST_DATASET_HPP

#include <CubbyDNN/Core/Memory.hpp>
#include <CubbyDNN/Datas/Dataset.hpp>
#include <CubbyDNN/Datas/SimpleData.hpp>

class TestDataset
    : public CubbyDNN::Dataset<
          TestDataset, CubbyDNN::SimpleData<CubbyDNN::Core::Memory<float>, CubbyDNN::Core::Memory<float>>>
{
 public:
    using OutputType = CubbyDNN::SimpleData<CubbyDNN::Core::Memory<float>, CubbyDNN::Core::Memory<float>>;

    OutputType Get(std::size_t index) override
    {
        using CubbyDNN::Core::Memory;

        Memory<float> data(4);
        Memory<float> target(5);

        data.GetSpan().FillScalar(index + 1.f);
        target.GetSpan()[index] = 1;

        return { data, target };
    }

    std::size_t GetSize() const override
    {
        return 5;
    }
};

#endif  // CUBBYDNN_TEST_TEST_DATASET_HPP
