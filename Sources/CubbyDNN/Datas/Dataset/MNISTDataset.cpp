#include <CubbyDNN/Datas/Dataset/MNISTDataset.hpp>

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
MNISTDataset::MNISTDataset(const std::string& root, bool train)
    : m_isTrain(train)
{
    if (train)
    {
        loadImages(root + "/train-images.idx3-ubyte");
        loadLabels(root + "/train-labels.idx1-ubyte");
    }
    else
    {
        loadImages(root + "/t10k-images.idx3-ubyte");
        loadLabels(root + "/t1ok-labels.idx1-ubyte");
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

std::size_t MNISTDataset::Size() const
{
    return m_images.size();
}

void MNISTDataset::loadImages(const std::string& filename)
{
    std::ifstream file(filename,
                       std::ios::in | std::ios::binary | std::ios::ate);

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
    std::ifstream file(filename,
                       std::ios::in | std::ios::binary | std::ios::ate);

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
