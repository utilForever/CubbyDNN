#ifndef CUBBYDNN_MNIST_DATASET_HPP
#define CUBBYDNN_MNIST_DATASET_HPP

#include <CubbyDNN/Datas/Image.hpp>

#include <optional>
#include <string>
#include <tuple>

namespace CubbyDNN
{
class MNISTDataset
{
 public:
    MNISTDataset(const std::string& root, bool train);

    bool IsTrain() const;
    std::tuple<Image, std::uint32_t> Get(std::size_t index) const;
    std::optional<std::size_t> Size() const;

 private:
    void loadImages(const std::string& filename);
    void loadLabels(const std::string& filename);

    std::uint32_t m_rowSize{ 0 }, m_colSize{ 0 };
    bool m_isTrain, m_loaded{ false };
    std::vector<Image> m_images;
    std::vector<int> m_labels;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_MNIST_DATASET_HPP
