#ifndef CUBBYDNN_TEST_TEST_DATASET_HPP
#define CUBBYDNN_TEST_TEST_DATASET_HPP

#include <CubbyDNN/Datas/Dataset.hpp>
#include <CubbyDNN/Datas/SimpleData.hpp>
#include <CubbyDNN/Datas/Tensor.hpp>

class TestDataset
    : public CubbyDNN::Dataset<
          TestDataset, CubbyDNN::SimpleData<CubbyDNN::FloatTensor, long>>
{
 public:
    OutputType Get(std::size_t index) override
    {
        static CubbyDNN::SimpleData<CubbyDNN::FloatTensor, long> arr[] = {
            { { 1, 1, 1, 1 }, 10 },
            { { 2, 2, 2, 2 }, 20 },
            { { 3, 3, 3, 3 }, 30 },
            { { 4, 4, 4, 4 }, 40 },
            { { 5, 5, 5, 5 }, 50 }
        };

        return arr[index];
    }

    std::size_t GetSize() const override
    {
        return 5;
    }
};

#endif  // CUBBYDNN_TEST_TEST_DATASET_HPP
