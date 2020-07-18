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

    virtual OutputType Get(std::size_t index) = 0;
    std::vector<OutputType> GetBatch(const std::vector<std::size_t>& indicies)
    {
        std::vector<OutputType> batch(indicies.size());

        for (std::size_t b = 0; b < indicies.size(); ++b)
            batch[b] = Get(indicies[b]);

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
