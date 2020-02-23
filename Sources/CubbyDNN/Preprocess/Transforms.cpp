#include <CubbyDNN/Preprocess/Transforms.hpp>

#include <cmath>
#include <stdexcept>

namespace
{
constexpr double PI = 3.141592653589793238462643383279;
}

namespace CubbyDNN::Transforms
{
CenterCrop::CenterCrop(std::size_t size) : m_cropSize_(size)
{
}

Image CenterCrop::operator()(const Image& origin)
{
    if (origin.GetWidth() < m_cropSize_ || origin.GetHeight())
    {
        throw std::invalid_argument(
            "Crop size is bigger than original image's size");
    }

    Image result(m_cropSize_, m_cropSize_, origin.HasAlpha(), origin.IsGrayScale());

    const std::size_t startX = origin.GetWidth() / 2 - m_cropSize_ / 2;
    const std::size_t startY = origin.GetHeight() / 2 - m_cropSize_ / 2;

    for (std::size_t y = 0; y < m_cropSize_; ++y)
    {
        for (std::size_t x = 0; x < m_cropSize_; ++x)
        {
            result.At(x, y) = origin.At(startX + x, startY + y);
        }
    }

    return result;
}

Image FlipHorizontal::operator()(const Image& origin)
{
    Image result(origin.GetWidth(), origin.GetHeight(), origin.HasAlpha());

    for (std::size_t y = 0; y < origin.GetHeight(); ++y)
    {
        for (std::size_t x = 0; x < origin.GetWidth(); ++x)
        {
            result.At(x, y) = origin.At(origin.GetWidth() - x - 1, y);
        }
    }

    return result;
}

Image FlipVertical::operator()(const Image& origin)
{
    Image result(origin.GetWidth(), origin.GetHeight(), origin.HasAlpha());

    for (std::size_t y = 0; y < origin.GetHeight(); ++y)
    {
        for (std::size_t x = 0; x < origin.GetWidth(); ++x)
        {
            result.At(x, y) = origin.At(x, origin.GetHeight() - y - 1);
        }
    }

    return result;
}

Rotation::Rotation(double degree) : m_rotationDegree_(degree)
{    
}

Image Rotation::operator()(const Image& origin)
{
    Image result(origin.GetWidth(), origin.GetHeight(), origin.HasAlpha());

    const double cosV = std::cos(PI * m_rotationDegree_ / 180.);
    const double sinV = std::sin(PI * m_rotationDegree_ / 180.);
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

Image GrayScale::operator()(const Image& origin)
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
}  // namespace CubbyDNN::Transforms
