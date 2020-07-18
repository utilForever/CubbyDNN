#ifndef CUBBYDNN_TEST_TEST_DATASET_HPP
#define CUBBYDNN_TEST_TEST_DATASET_HPP

#include <CubbyDNN/Datas/Dataset.hpp>
#include <CubbyDNN/Datas/Tensor.hpp>

class TestDataset : public CubbyDNN::Dataset<TestDataset, std::tuple<CubbyDNN::FloatTensor, long>>
{
 public:
    OutputType Get(std::size_t index) override
    {
        static std::tuple<CubbyDNN::FloatTensor, long> arr[] = {
            std::make_tuple<CubbyDNN::FloatTensor, long>({ 1, 1, 1, 1 }, 10),
            std::make_tuple<CubbyDNN::FloatTensor, long>({ 2, 2, 2, 2 }, 20),
            std::make_tuple<CubbyDNN::FloatTensor, long>({ 3, 3, 3, 3 }, 30),
            std::make_tuple<CubbyDNN::FloatTensor, long>({ 4, 4, 4, 4 }, 40),
            std::make_tuple<CubbyDNN::FloatTensor, long>({ 5, 5, 5, 5 }, 50)
        };

        return arr[index];
    }

    std::size_t GetSize() const override
    {
        return 5;
    }
};

#endif  // CUBBYDNN_TEST_TEST_DATASET_HPP
