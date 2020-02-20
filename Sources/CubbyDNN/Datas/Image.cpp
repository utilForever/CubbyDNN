#include <CubbyDNN/Datas/Image.hpp>

#include <stdexcept>

namespace CubbyDNN
{
Pixel::Pixel(unsigned char r, unsigned char g, unsigned char b)
    : m_r(r), m_g(g), m_b(b)
{
}

Pixel::Pixel(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
    : m_r(r), m_g(g), m_b(b), m_a(a)
{
}

Pixel::Pixel(unsigned char gray) : m_a(gray), m_grayScale(true)
{
}

unsigned char& Pixel::R()
{
    return m_r;
}

unsigned char Pixel::R() const
{
    return m_r;
}

unsigned char& Pixel::G()
{
    return m_g;
}

unsigned char Pixel::G() const
{
    return m_g;
}

unsigned char& Pixel::B()
{
    return m_b;
}

unsigned char Pixel::B() const
{
    return m_b;
}

unsigned char& Pixel::A()
{
    return m_a;
}

unsigned char Pixel::A() const
{
    return m_a;
}

unsigned char& Pixel::Gray()
{
    if (!m_grayScale)
        throw std::runtime_error("Pixel is not gray scale");

    return m_a;
}

unsigned char Pixel::Gray() const
{
    if (!m_grayScale)
        throw std::runtime_error("Pixel is not gray scale");

    return m_a;
}

void Pixel::ToGrayScale()
{
    m_a = static_cast<unsigned char>(0.299 * m_r + 0.587 * m_g + 0.144 * m_b);

    m_grayScale = true;
}

bool Pixel::operator==(const Pixel& other) const
{
    return (m_a == other.m_a) && (m_r == other.m_r) && (m_g == other.m_g) &&
           (m_b == other.m_b);
}

bool Pixel::operator!=(const Pixel& other) const
{
    return !(*this == other);
}

Image::Image(std::size_t width, std::size_t height, bool hasAlpha,
             bool grayScale)
    : m_width(width),
      m_height(height),
      m_hasAlpha(hasAlpha),
      m_grayScale(grayScale),
      m_data(width * height)
{
    if (grayScale)
    {
        for (auto& pixel : m_data)
        {
            pixel.ToGrayScale();
        }
    }
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

Pixel& Image::At(std::size_t x, std::size_t y)
{
    return const_cast<Pixel&>(std::as_const(*this).At(x, y));
}

const Pixel& Image::At(std::size_t x, std::size_t y) const
{
    return m_data[x + y * m_width];
}
}  // namespace CubbyDNN
