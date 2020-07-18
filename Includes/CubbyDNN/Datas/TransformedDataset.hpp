#ifndef CUBBYDNN_TRANSFORMED_DATASET_HPP
#define CUBBYDNN_TRANSFORMED_DATASET_HPP

#include <CubbyDNN/Datas/Dataset.hpp>

namespace CubbyDNN
{
template <class DatasetT, class TransformT>
class TransformedDataset final
    : public Dataset<TransformedDataset<DatasetT, TransformT>,
                     typename TransformT::OutputType>
{
 public:
    using InputType = typename TransformT::InputType;
    using OutputType = typename TransformT::OutputType;

    TransformedDataset(DatasetT dataset, TransformT transform)
        : m_dataset(std::move(dataset)), m_transform(std::move(transform))
    {
    }

    OutputType Get(std::size_t index) override
    {
        return m_transform(m_dataset.Get(index));
    }

    std::size_t GetSize() const
    {
        return m_dataset.GetSize();
    }

 private:
    DatasetT m_dataset;
    TransformT m_transform;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TRANSFORMED_DATASET_HPP
