#include <cubbydnn/Datas/Image.hpp>

#include <fstream>
#include <stdexcept>

namespace CubbyDNN
{
bool Image::Pixel::operator==(const Pixel& other) const
{
    return (r == other.r) && (g == other.g) && (b == other.b);
}

bool Image::Pixel::operator!=(const Pixel& other) const
{
    return !(*this == other);
}

Image::Image(std::size_t width, std::size_t height)
    : m_width(width), m_height(height), m_data(width * height)
{
}

std::size_t Image::GetWidth() const
{
    return m_width;
}

std::size_t Image::GetHeight() const
{
    return m_height;
}

Image Image::Load(const std::string& filename)
{
    switch (checkFileFormat(filename))
    {
        case FileFormat::BMP:
            return loadBMP(filename);
        default:
            throw std::runtime_error("Unsupported file format");
    }
}

void Image::Save(const std::string& filename, FileFormat format)
{
    (void)filename;
    (void)format;

    throw std::runtime_error("Not implemented");
}

Image::Pixel& Image::At(std::size_t x, std::size_t y)
{
    return const_cast<Pixel&>(std::as_const(*this).At(x, y));
}

const Image::Pixel& Image::At(std::size_t x, std::size_t y) const
{
    return m_data[x + y * m_width];
}

Image::FileFormat Image::checkFileFormat(const std::string& filename)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);

    Image result;

    std::string header;
    header.resize(2);

    file.read(header.data(), 2);

    if (header[0] == 0x42 || header[1] == 0x4D)
    {
        file.close();
        return FileFormat::BMP;
    }

    file.close();
    return FileFormat::UNKNOWN;
}

Image Image::loadBMP(const std::string& filename)
{
    std::basic_ifstream<unsigned char> file(filename,
                                            std::ios::in | std::ios::binary);

    unsigned char header[54];
    file.read(header, 54);

    const std::size_t width = *reinterpret_cast<int*>(&header[18]);
    const std::size_t height = *reinterpret_cast<int*>(&header[22]);

    Image image(width, height);

    const int rowPadded = (3 * width + 3) & (~3);
    std::vector<unsigned char> data(rowPadded);

    for (std::size_t y = 0; y < height; ++y)
    {
        file.read(data.data(), rowPadded);

        for (std::size_t x = 0; x < width; ++x)
        {
            image.At(x, y).r = data[3 * x + 2];
            image.At(x, y).g = data[3 * x + 1];
            image.At(x, y).b = data[3 * x + 0];
        }
    }

    file.close();

    return image;
}
}  // namespace CubbyDNN
