#ifndef CUBBYDNN_IMAGE_TRANSFORMS_HPP
#define CUBBYDNN_IMAGE_TRANSFORMS_HPP

#include <CubbyDNN/Datas/Image.hpp>
#include <CubbyDNN/Datas/Tensor.hpp>
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

class RandomFlipHorizontal : public Transform<Image, Image>
{
 public:
    RandomFlipHorizontal(double p = 0.5);

    OutputType operator()(const InputType& input) override;

 private:
    double p_;
};

class RandomFlipVertical : public Transform<Image, Image>
{
 public:
    RandomFlipVertical(double p = 0.5);

    OutputType operator()(const Image& input) override;

 private:
    double p_;
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

class RandomGrayScale : public Transform<Image, Image>
{
public:
    RandomGrayScale(double p = 0.5);

    OutputType operator()(const InputType& input) override;

private:
    double p_;
};

class ToTensor : public Transform<Image, FloatTensor>
{
 public:
    OutputType operator()(const InputType& input) override;
};
}  // namespace CubbyDNN::Transforms::ImageTransforms

#endif  // CUBBYDNN_IMAGE_TRANSFORMS_HPP
