#include <CubbyDNN/Preprocess/ImageTransforms.hpp>

#include <cmath>
#include <stdexcept>

namespace
{
constexpr double PI = 3.141592653589793238462643383279;
}

namespace CubbyDNN::Transforms::ImageTransforms
{
CenterCrop::CenterCrop(std::size_t size) : m_cropSize_(size)
{
}

CenterCrop::OutputType CenterCrop::operator()(const InputType& origin)
{
    if (origin.GetWidth() < m_cropSize_ || origin.GetHeight())
    {
        throw std::invalid_argument(
            "Crop size is bigger than original image's size");
    }

    Image result(m_cropSize_, m_cropSize_, origin.HasAlpha(),
                 origin.IsGrayScale());

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

FlipHorizontal::OutputType FlipHorizontal::operator()(const InputType& origin)
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

FlipVertical::OutputType FlipVertical::operator()(const InputType& origin)
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

Rotation::OutputType Rotation::operator()(const InputType& origin)
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

GrayScale::OutputType GrayScale::operator()(const InputType& origin)
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

ToTensor::OutputType ToTensor::operator()(const InputType& input)
{
    const std::size_t heightSize = input.GetHeight();
    const std::size_t imageSize = input.GetHeight() * input.GetWidth();
    const std::size_t channelSize =
        (input.IsGrayScale() ? 1 : (input.HasAlpha() ? 4 : 3));

    std::vector<float> data(imageSize * channelSize);

    const auto index = [&imageSize, &heightSize](std::size_t x, std::size_t y,
                                                 std::size_t c) -> std::size_t {
        return c * imageSize + y * heightSize + x;
    };

    if (input.IsGrayScale())
    {
        for (std::size_t y = 0; y < input.GetHeight(); ++y)
        {
            for (std::size_t x = 0; x < input.GetWidth(); ++x)
            {
                const auto& pixel = input.At(x, y);
                data[index(x, y, 0)] = pixel.Gray() / 255.f;
            }
        }
    }
    else
    {
        for (std::size_t y = 0; y < input.GetHeight(); ++y)
        {
            for (std::size_t x = 0; x < input.GetWidth(); ++x)
            {
                const auto& pixel = input.At(x, y);
                data[index(x, y, 0)] = pixel.R() / 255.f;
                data[index(x, y, 1)] = pixel.G() / 255.f;
                data[index(x, y, 2)] = pixel.B() / 255.f;

                if (channelSize == 4)
                    data[index(x, y, 3)] = pixel.A() / 255.f;
            }
        }
    }

    return data;
}
}  // namespace CubbyDNN::Transforms::ImageTransforms
