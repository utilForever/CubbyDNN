#ifndef CUBBYDNN_TRANSFORMED_DATASET_HPP
#define CUBBYDNN_TRANSFORMED_DATASET_HPP

#include <CubbyDNN/Datas/Dataset.hpp>

namespace CubbyDNN
{
template <class DatasetT, class TransformT>
class TransformedDataset final : public Dataset<typename TransformT::OutputType>
{
 public:
    using InputType = typename TransformT::InputType;
    using OutputType = typename TransformT::OutputType;

    TransformedDataset(DatasetT dataset, TransformT transform)
        : m_dataset(std::move(dataset)), m_transform(std::move(transform))
    {
    }

    OutputType Get(std::size_t index) const override
    {
        return m_transform(m_dataset.Get(index));
    }

 private:
    DatasetT m_dataset;
    TransformT m_transform;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TRANSFORMED_DATASET_HPP
