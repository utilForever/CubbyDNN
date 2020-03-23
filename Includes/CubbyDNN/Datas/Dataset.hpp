#ifndef CUBBYDNN_DATASET_HPP
#define CUBBYDNN_DATASET_HPP

#include <tuple>
#include <vector>

namespace CubbyDNN
{
template <class DT, class TT>
class TransformedDataset;

template <class Self, class OutputT>
class Dataset
{
 public:
    using OutputType = OutputT;

    virtual ~Dataset() = default;

    virtual OutputType Get(std::size_t index) const = 0;
    std::vector<OutputType> Get(const std::vector<std::size_t>& indicies) const
    {
        std::vector<OutputType> batch(indicies.size());

        for (const std::size_t i : indicies)
            batch.push_back(Get(i));

        return batch;
    }

    virtual std::size_t GetSize() const = 0;

    template <class TransformT>
    TransformedDataset<Self, TransformT> Transform(TransformT transform)
    {
        return TransformedDataset<Self, TransformT>(std::move(*this), std::move(transform));
    }
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_DATASET_HPP
