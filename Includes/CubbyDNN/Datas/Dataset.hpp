#ifndef CUBBYDNN_DATASET_HPP
#define CUBBYDNN_DATASET_HPP

#include <tuple>
#include <vector>

namespace CubbyDNN
{
template <class OutputT>
class Dataset
{
 public:
    using OutputType = OutputT;

    virtual ~Dataset() = default;

    virtual OutputType Get(std::size_t index) const = 0;
    std::vector<OutputType> Get(const std::vector<std::size_t> indicies)
    {
        std::vector<OutputType> batch;

        for (const std::size_t i : indicies)
            batch.push_back(Get(i));

        return batch;
    }

    virtual std::size_t GetSize() const = 0;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_DATASET_HPP
