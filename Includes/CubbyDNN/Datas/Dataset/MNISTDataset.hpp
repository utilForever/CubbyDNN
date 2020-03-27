#ifndef CUBBYDNN_MNIST_DATASET_HPP
#define CUBBYDNN_MNIST_DATASET_HPP

#include <CubbyDNN/Datas/Dataset.hpp>
#include <CubbyDNN/Datas/Image.hpp>

#include <string>
#include <tuple>

namespace CubbyDNN
{
class MNISTDataset final : public Dataset<MNISTDataset, std::tuple<Image, long>>
{
 public:
    MNISTDataset(const std::string& root, bool train);

    bool IsTrain() const;
    OutputType Get(std::size_t index) const override;
    std::size_t GetSize() const override;

 private:
    void loadImages(const std::string& filename);
    void loadLabels(const std::string& filename);

    std::uint32_t m_rowSize{ 0 }, m_colSize{ 0 };
    bool m_isTrain, m_loaded{ false };
    std::vector<Image> m_images;
    std::vector<long> m_labels;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_MNIST_DATASET_HPP
