#include <CubbyDNN/Datas/Image.hpp>

namespace CubbyDNN
{
bool Image::Pixel::operator==(const Pixel& other) const
{
    return (a == other.a) && (r == other.r) && (g == other.g) && (b == other.b);
}

bool Image::Pixel::operator!=(const Pixel& other) const
{
    return !(*this == other);
}

Image::Image(std::size_t width, std::size_t height, bool hasAlpha)
    : m_width(width), m_height(height), m_data(width * height), m_hasAlpha(hasAlpha)
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

bool Image::HasAlpha() const
{
    return m_hasAlpha;
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
