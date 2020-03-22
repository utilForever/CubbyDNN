#ifndef CUBBYDNN_IMAGE_TRANSFORMS_HPP
#define CUBBYDNN_IMAGE_TRANSFORMS_HPP

#include <CubbyDNN/Datas/Image.hpp>
#include <CubbyDNN/Preprocess/Transforms.hpp>

namespace CubbyDNN::Transforms::ImageTransforms
{
class CenterCrop : public Transform<Image, Image>
{
 public:
    CenterCrop(std::size_t size);

    OutputType operator()(const InputType& input) override;

 private:
    std::size_t m_cropSize_;
};

class FlipHorizontal : public Transform<Image, Image>
{
 public:
    OutputType operator()(const InputType& input) override;
};

class FlipVertical : public Transform<Image, Image>
{
 public:
    OutputType operator()(const Image& input) override;
};

class Rotation : public Transform<Image, Image>
{
 public:
    Rotation(double degree);

    OutputType operator()(const InputType& input) override;

 private:
    double m_rotationDegree_;
};

class GrayScale : public Transform<Image, Image>
{
 public:
    OutputType operator()(const InputType& input) override;
};

class ToTensor : public Transform<Image, std::vector<float>>
{
 public:
    OutputType operator()(const InputType& input) override
    {
        const std::size_t heightSize = input.GetHeight();
        const std::size_t imageSize = input.GetHeight() * input.GetWidth();
        const std::size_t channelSize =
            (input.IsGrayScale() ? 1 : (input.HasAlpha() ? 4 : 3));

        std::vector<float> data(imageSize * channelSize);

        const auto index = [&imageSize, &heightSize](
                               std::size_t x, std::size_t y,
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
                        data[index(x, y, 3)] =
                            pixel.A() / 255.f;
                }
            }
        }
    }
};
}  // namespace CubbyDNN::Transforms::ImageTransforms

#endif  // CUBBYDNN_IMAGE_TRANSFORMS_HPP
