#ifndef CUBBYDNN_DATASET_HPP
#define CUBBYDNN_DATASET_HPP

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

    virtual std::size_t GetSize() const = 0;

    template <class TransformT>
    TransformedDataset<Self, TransformT> Transform(TransformT transform) &
    {
        return TransformedDataset<Self, TransformT>{ static_cast<Self&>(*this),
                                                     std::move(transform) };
    }

    template <class TransformT>
    TransformedDataset<Self, TransformT> Transform(TransformT transform) &&
    {
        return TransformedDataset<Self, TransformT>{
            std::move(static_cast<Self&>(*this)), std::move(transform)
        };
    }
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_DATASET_HPP
