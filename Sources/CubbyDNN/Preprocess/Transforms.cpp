#include <CubbyDNN/Preprocess/Transforms.hpp>

#include <cmath>

namespace
{
constexpr double PI = 3.141592653589793238462643383279;
}

namespace CubbyDNN::Transforms
{
Image FlipHorizontal(const Image& origin)
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

Image FlipVertical(const Image& origin)
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

Image Rotation(const Image& origin, double degree)
{
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

Image GrayScale(const Image& origin)
{
    Image result(origin);

    for (std::size_t y = 0; y < result.GetHeight(); ++y)
    {
        for (std::size_t x = 0; x < result.GetWidth(); ++x)
        {
            result.At(x, y).ToGrayScale();
        }
    }

    return result;
}
}  // namespace CubbyDNN::Transforms
