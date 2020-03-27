#ifndef CUBBYDNN_DATA_LOADER_HPP
#define CUBBYDNN_DATA_LOADER_HPP

#include <CubbyDNN/Datas/Tensor.hpp>

#include <algorithm>
#include <effolkronium/random.hpp>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>

namespace CubbyDNN
{
template <class Dataset>
class DataLoader
{
 public:
    using Batch = std::tuple<FloatTensor, LongTensor>;

    DataLoader(std::unique_ptr<Dataset> dataset, std::size_t batchSize = 1,
               bool shuffle = false)
        : m_dataset(std::move(dataset)),
          m_batchSize(batchSize),
          m_shuffle(shuffle)
    {
        if (m_dataset->GetSize() < batchSize)
        {
            throw std::invalid_argument(
                "Batch size cannot bigger than dataset size");
        }

        m_datasetSize = m_dataset->GetSize();
        m_datasetSize -= m_datasetSize % batchSize;  // Drop Last

        m_indices.resize(m_datasetSize);
    }

    void Begin();
    std::optional<Batch> Next();

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
    std::unique_ptr<Dataset> m_dataset;

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
std::optional<typename DataLoader<Dataset>::Batch> DataLoader<Dataset>::Next()
{
    if (m_nowPos >= m_datasetSize)
        return std::nullopt;

    std::vector<std::size_t> indices(m_batchSize);
    for (std::size_t dataIdx = m_nowPos, batch = 0;
         dataIdx < m_nowPos + m_batchSize; ++dataIdx, ++batch)
        indices[batch] = m_indices[dataIdx];

    const auto batch = m_dataset->GetBatch(indices);

    const std::size_t inputSize = std::get<0>(batch[0]).size();

    FloatTensor input(m_batchSize * inputSize);
    LongTensor target(m_batchSize);

    for (std::size_t batchIdx = 0; batchIdx < m_batchSize; ++batchIdx)
    {
        auto [inp, tar] = batch[batchIdx];

        std::copy(begin(inp), end(inp), begin(input) + inputSize * batchIdx);
        target[batchIdx] = tar;
    }

    m_nowPos += m_batchSize;

    return std::make_optional(std::make_tuple(input, target));
}
}  // namespace CubbyDNN

#endif  // CUBBYDNN_DATA_LOADER_HPP
