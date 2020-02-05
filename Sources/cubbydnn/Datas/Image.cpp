#include <cubbydnn/Datas/Image.hpp>

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

std::size_t Image::GetWidth() const
{
    return m_width;
}

std::size_t Image::GetHeight() const
{
    return m_height;
}

Image::Pixel& Image::At(std::size_t x, std::size_t y)
{
    return const_cast<Pixel&>(std::as_const(*this).At(x, y));
}

const Image::Pixel& Image::At(std::size_t x, std::size_t y) const
{
    return m_data[x + y * m_width];
}
}  // namespace CubbyDNN
