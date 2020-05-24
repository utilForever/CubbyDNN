#include <CubbyDNN/Datas/Image.hpp>

#include <cmath>
#include <stdexcept>
#include <utility>

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
    m_a = ToGrayScale(*this).Gray();

    m_grayScale = true;
}

Pixel Pixel::ToGrayScale(const Pixel& pixel)
{
    return Pixel(static_cast<unsigned char>(
        0.299 * pixel.R() + 0.587 * pixel.G() + 0.144 * pixel.B()));
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

Image Image::ToGrayScale(const Image& origin)
{
    Image result(origin.GetWidth(), origin.GetHeight(), false, true);

    for (std::size_t y = 0; y < result.GetHeight(); ++y)
    {
        for (std::size_t x = 0; x < result.GetWidth(); ++x)
        {
            result.At(x, y) = Pixel::ToGrayScale(origin.At(x, y));
        }
    }

    return result;
}

Image Image::Rotate(const Image& origin, double degree)
{
    constexpr double PI = 3.141592653589793238462643383279;

    Image result(origin.GetWidth(), origin.GetHeight(), origin.HasAlpha());

    const double cosV = std::cos(PI * degree / 180.);
    const double sinV = std::sin(PI * degree / 180.);
    const double centerX = origin.GetWidth() / 2.,
                 centerY = origin.GetHeight() / 2.;

    for (std::size_t y = 0; y < origin.GetHeight(); ++y)
    {
        for (std::size_t x = 0; x < origin.GetWidth(); ++x)
        {
            const double origX =
                (centerX + (y - centerY) * sinV + (x - centerX) * cosV);
            const double origY =
                (centerY + (y - centerY) * cosV - (x - centerX) * sinV);

            if ((origX >= 0 &&
                 static_cast<std::size_t>(origX) < origin.GetWidth()) &&
                (origY >= 0 &&
                 static_cast<std::size_t>(origY) < origin.GetHeight()))
                result.At(x, y) = origin.At(static_cast<std::size_t>(origX),
                                            static_cast<std::size_t>(origY));
        }
    }

    return result;
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

bool Image::IsGrayScale() const
{
    return m_grayScale;
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
