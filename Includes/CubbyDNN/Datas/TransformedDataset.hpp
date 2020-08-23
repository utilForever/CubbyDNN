#ifndef CUBBYDNN_TRANSFORMED_DATASET_HPP
#define CUBBYDNN_TRANSFORMED_DATASET_HPP

#include <CubbyDNN/Datas/Dataset.hpp>
#include <CubbyDNN/Datas/SimpleData.hpp>

namespace CubbyDNN
{
template <class DatasetT, class TransformT>
class TransformedDataset final
    : public Dataset<TransformedDataset<DatasetT, TransformT>,
                     SimpleData<typename TransformT::OutputType,
                                typename DatasetT::OutputType::OutputType>>
{
 public:
    using
        typename Dataset<TransformedDataset<DatasetT, TransformT>,
                         SimpleData<typename TransformT::OutputType,
                                    typename DatasetT::OutputType::OutputType>>::OutputType;

    TransformedDataset(DatasetT dataset, TransformT transform)
        : m_dataset(std::move(dataset)), m_transform(std::move(transform))
    {
    }

    OutputType Get(std::size_t index) override
    {
        auto data = m_dataset.Get(index);

        OutputType ret;
        ret.Data = m_transform(data.Data);
        Core::Swap(ret.Target, data.Target);

        return ret;
    }

    std::size_t GetSize() const override
    {
        return m_dataset.GetSize();
    }

 private:
    DatasetT m_dataset;
    TransformT m_transform;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TRANSFORMED_DATASET_HPP
