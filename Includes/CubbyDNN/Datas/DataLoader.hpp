#ifndef CUBBYDNN_DATA_LOADER_HPP
#define CUBBYDNN_DATA_LOADER_HPP

#include <CubbyDNN/Core/Memory.hpp>
#include <CubbyDNN/Datas/SimpleData.hpp>

#include <algorithm>
#include <effolkronium/random.hpp>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

namespace CubbyDNN
{
template <class Dataset>
class DataLoader
{
 public:
    DataLoader(Dataset dataset, std::size_t batchSize = 1,
               bool shuffle = false)
        : m_dataset(std::move(dataset)),
          m_batchSize(batchSize),
          m_shuffle(shuffle)
    {
        if (m_dataset.GetSize() < batchSize)
        {
            throw std::invalid_argument(
                "Batch size cannot bigger than dataset size");
        }

        m_datasetSize = m_dataset.GetSize();
        m_datasetSize -= m_datasetSize % batchSize;  // Drop Last

        m_indices.resize(m_datasetSize);
    }

    void Begin();
    std::optional<SimpleBatch> Next();

    std::size_t GetSize() const
    {
        return m_datasetSize / m_batchSize;
    }

    std::size_t GetBatchSize() const
    {
        return m_batchSize;
    }

    bool DoShuffle() const
    {
        return m_shuffle;
    }

 private:
    Dataset m_dataset;

    std::size_t m_batchSize;
    std::size_t m_datasetSize;
    bool m_shuffle;

    std::vector<std::size_t> m_indices;
    std::size_t m_nowPos{ 0 };
};

template <class Dataset>
void DataLoader<Dataset>::Begin()
{
    m_nowPos = 0;
    std::iota(begin(m_indices), end(m_indices), 0);

    if (m_shuffle)
    {
        effolkronium::random_static::shuffle(m_indices);
    }
}

template <class Dataset>
std::optional<SimpleBatch> DataLoader<Dataset>::Next()
{
    static_assert(
        std::is_same<Core::Memory<float>,
                     typename Dataset::OutputType::InputType>::value == true);
    static_assert(
        std::is_same<Core::Memory<float>,
                     typename Dataset::OutputType::OutputType>::value == true);

    if (m_nowPos >= m_datasetSize)
        return std::nullopt;

    std::vector<std::size_t> indices(m_batchSize);
    for (std::size_t dataIdx = m_nowPos, batch = 0;
         dataIdx < m_nowPos + m_batchSize; ++dataIdx, ++batch)
        indices[batch] = m_indices[dataIdx];

    m_nowPos += m_batchSize;

    std::vector<typename Dataset::OutputType> batches(m_batchSize);
    std::size_t dataSize = 0, targetSize = 0;
    for (std::size_t b = 0; b < m_batchSize; ++b)
    {
        batches[b] = m_dataset.Get(indices[b]);

        dataSize += batches[b].Data.Size();
        targetSize += batches[b].Target.Size();
    }

    Core::Memory<float> data(dataSize);
    Core::Memory<float> target(targetSize);

    std::size_t dataPos = 0, targetPos = 0;
    for (std::size_t b = 0; b < m_batchSize; ++b)
    {
        const std::size_t dataLength = batches[b].Data.Size();
        const std::size_t targetLength = batches[b].Target.Size();

        std::copy(batches[b].Data.GetSpan().begin(),
                  batches[b].Data.GetSpan().end(),
                  data.GetSpan().begin() + dataPos);
        std::copy(batches[b].Target.GetSpan().begin(),
                  batches[b].Target.GetSpan().end(),
                  target.GetSpan().begin() + targetPos);

        dataPos += dataLength;
        targetPos += targetLength;
    }

    return SimpleBatch{ data, target };
}
}  // namespace CubbyDNN

#endif  // CUBBYDNN_DATA_LOADER_HPP
