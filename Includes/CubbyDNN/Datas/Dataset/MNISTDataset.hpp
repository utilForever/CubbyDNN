#ifndef CUBBYDNN_MNIST_DATASET_HPP
#define CUBBYDNN_MNIST_DATASET_HPP

#include <CubbyDNN/Core/Memory.hpp>
#include <CubbyDNN/Datas/Dataset.hpp>
#include <CubbyDNN/Datas/Image.hpp>
#include <CubbyDNN/Datas/SimpleData.hpp>

#include <string>

namespace CubbyDNN
{
class MNISTDataset final : public Dataset<MNISTDataset, SimpleData<Image, Core::Memory<float>>>
{
 public:
    MNISTDataset(const std::string& root, bool train, bool download);

    bool IsTrain() const;
    OutputType Get(std::size_t index) override;
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
