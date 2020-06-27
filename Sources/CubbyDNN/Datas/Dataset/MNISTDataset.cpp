#include <CubbyDNN/Datas/Dataset/MNISTDataset.hpp>

#include <CubbyDNN/Utils/Downloader.hpp>

#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#endif
#include <fstream>

namespace
{
bool checkIsLittleEndian()
{
    const uint32_t word = 1;
    return reinterpret_cast<const uint8_t*>(&word)[0] == 1;
}

constexpr std::uint32_t flipEndianess(std::uint32_t value)
{
    return ((value & 0xffu) << 24u) | ((value & 0xff00u) << 8u) |
           ((value & 0xff0000u) >> 8u) | ((value & 0xff000000u) >> 24u);
}

std::uint32_t readInt32(std::ifstream& stream)
{
    static const bool isLittleEndian = checkIsLittleEndian();

    uint32_t value;
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));

    return isLittleEndian ? flipEndianess(value) : value;
}
}  // namespace

namespace CubbyDNN
{
MNISTDataset::MNISTDataset(const std::string& root, bool train, bool download)
    : m_isTrain(train)
{
    const std::string train_images_path = root + "/train-images.idx3-ubyte";
    const std::string train_labels_path = root + "/train-labels.idx1-ubyte";
    const std::string test_images_path = root + "/t10k-images.idx3-ubyte";
    const std::string test_labels_path = root + "/t10k-labels.idx1-ubyte";

    if (download)
    {
        if (!filesystem::exists(root))
        {
            filesystem::create_directories(root);
        }

        std::ofstream train_images(train_images_path + ".gz");
        if (!Downloader::DownloadData(
                "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                train_images))
            throw std::runtime_error("Cannot download train images");
        train_images.close();
        if (!Downloader::UnGzip(train_images_path + ".gz", train_images_path))
            throw std::runtime_error("Cannot extract train images");

        std::ofstream train_labels(train_labels_path + ".gz");
        if (!Downloader::DownloadData(
                "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                train_labels))
            throw std::runtime_error("Cannot download train labels");
        train_labels.close();
        if (!Downloader::UnGzip(train_labels_path + ".gz", train_labels_path))
            throw std::runtime_error("Cannot extract train labels");

        std::ofstream test_images(test_images_path + ".gz");
        if (!Downloader::DownloadData(
                "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                test_images))
            throw std::runtime_error("Cannot download test images");
        test_images.close();
        if (!Downloader::UnGzip(test_images_path + ".gz", test_images_path))
            throw std::runtime_error("Cannot extract test images");

        std::ofstream test_labels(test_labels_path + ".gz");
        if (!Downloader::DownloadData(
                "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                test_labels))
            throw std::runtime_error("Cannot download test labels");
        test_labels.close();
        if (!Downloader::UnGzip(test_labels_path + ".gz", test_labels_path))
            throw std::runtime_error("Cannot extract test labels");
    }

    if (train)
    {
        loadImages(train_images_path);
        loadLabels(train_labels_path);
    }
    else
    {
        loadImages(test_images_path);
        loadLabels(test_labels_path);
    }

    m_loaded = true;
}

bool MNISTDataset::IsTrain() const
{
    return m_isTrain;
}

MNISTDataset::OutputType MNISTDataset::Get(std::size_t index) const
{
    return { m_images[index], m_labels[index] };
}

std::size_t MNISTDataset::GetSize() const
{
    return m_images.size();
}

void MNISTDataset::loadImages(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file");
    }

    const uint32_t magicNumber = readInt32(file);

    if (magicNumber != 2051)
    {
        file.close();
        throw std::runtime_error("Invalid image file");
    }

    const uint32_t numOfImages = readInt32(file);
    m_rowSize = readInt32(file);
    m_colSize = readInt32(file);

    for (uint32_t i = 0; i < numOfImages; ++i)
    {
        Image image(m_rowSize, m_colSize, false, true);

        for (uint32_t y = 0; y < m_colSize; ++y)
        {
            for (uint32_t x = 0; x < m_rowSize; ++x)
            {
                file.read(reinterpret_cast<char*>(&image.At(x, y).Gray()), 1);
            }
        }

        m_images.emplace_back(image);
    }

    file.close();
}

void MNISTDataset::loadLabels(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file");
    }

    const uint32_t magicNumber = readInt32(file);

    if (magicNumber != 2049)
    {
        file.close();
        throw std::runtime_error("Invalid label file");
    }

    const uint32_t numOfItems = readInt32(file);

    for (uint32_t i = 0; i < numOfItems; ++i)
    {
        unsigned char value;
        file.read(reinterpret_cast<char*>(&value), 1);

        m_labels.emplace_back(value);
    }

    file.close();
}
}  // namespace CubbyDNN
