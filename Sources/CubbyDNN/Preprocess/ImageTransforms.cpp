#include <CubbyDNN/Preprocess/ImageTransforms.hpp>

#include <effolkronium/random.hpp>
#include <stdexcept>

namespace CubbyDNN::Transforms::ImageTransforms
{
CenterCrop::CenterCrop(std::size_t size) : m_cropSize_(size)
{
}

CenterCrop::OutputType CenterCrop::operator()(const InputType& input)
{
    if (input.GetWidth() < m_cropSize_ || input.GetHeight())
    {
        throw std::invalid_argument(
            "Crop size is bigger than inputal image's size");
    }

    Image result(m_cropSize_, m_cropSize_, input.HasAlpha(),
                 input.IsGrayScale());

    const std::size_t startX = input.GetWidth() / 2 - m_cropSize_ / 2;
    const std::size_t startY = input.GetHeight() / 2 - m_cropSize_ / 2;

    for (std::size_t y = 0; y < m_cropSize_; ++y)
    {
        for (std::size_t x = 0; x < m_cropSize_; ++x)
        {
            result.At(x, y) = input.At(startX + x, startY + y);
        }
    }

    return result;
}

RandomFlipHorizontal::RandomFlipHorizontal(double p) : m_prob(p)
{
}

RandomFlipHorizontal::OutputType RandomFlipHorizontal::operator()(
    const InputType& input)
{
    if (effolkronium::random_static::get<bool>(m_prob))
    {
        return input;
    }

    Image result(input.GetWidth(), input.GetHeight(), input.HasAlpha());

    for (std::size_t y = 0; y < input.GetHeight(); ++y)
    {
        for (std::size_t x = 0; x < input.GetWidth(); ++x)
        {
            result.At(x, y) = input.At(input.GetWidth() - x - 1, y);
        }
    }

    return result;
}

RandomFlipVertical::RandomFlipVertical(double p) : m_prob(p)
{
}

RandomFlipVertical::OutputType RandomFlipVertical::operator()(
    const InputType& input)
{
    if (effolkronium::random_static::get<bool>(m_prob))
    {
        return input;
    }

    Image result(input.GetWidth(), input.GetHeight(), input.HasAlpha());

    for (std::size_t y = 0; y < input.GetHeight(); ++y)
    {
        for (std::size_t x = 0; x < input.GetWidth(); ++x)
        {
            result.At(x, y) = input.At(x, input.GetHeight() - y - 1);
        }
    }

    return result;
}

RandomRotation::RandomRotation(double degree, double p)
    : m_rotationDegree(degree), m_prob(p)
{
}

RandomRotation::OutputType RandomRotation::operator()(const InputType& input)
{
    if (effolkronium::random_static::get<bool>(m_prob))
    {
        return input;
    }

    return Image::Rotate(input, effolkronium::random_static::get<bool>(0.5)
                                    ? m_rotationDegree
                                    : -m_rotationDegree);
}

GrayScale::OutputType GrayScale::operator()(const InputType& input)
{
    return Image::ToGrayScale(input);
}

RandomGrayScale::RandomGrayScale(double p) : m_prob(p)
{
}

Transform<Image, Image>::OutputType RandomGrayScale::operator()(
    const InputType& input)
{
    if (effolkronium::random_static::get<bool>(m_prob))
    {
        return input;
    }

    return Image::ToGrayScale(input);
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
