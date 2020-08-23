#ifndef CUBBYDNN_IMAGE_TRANSFORMS_HPP
#define CUBBYDNN_IMAGE_TRANSFORMS_HPP

#include <CubbyDNN/Core/Memory.hpp>
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

class RandomFlipHorizontal : public Transform<Image, Image>
{
 public:
    RandomFlipHorizontal(double p = 0.5);

    OutputType operator()(const InputType& input) override;

 private:
    double m_prob;
};

class RandomFlipVertical : public Transform<Image, Image>
{
 public:
    RandomFlipVertical(double p = 0.5);

    OutputType operator()(const InputType& input) override;

 private:
    double m_prob;
};

class RandomRotation : public Transform<Image, Image>
{
 public:
    RandomRotation(double degree, double p = 0.5);

    OutputType operator()(const InputType& input) override;

 private:
    double m_rotationDegree;
    double m_prob;
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
    double m_prob;
};

class ToMemory : public Transform<Image, Core::Memory<float>>
{
 public:
    OutputType operator()(const InputType& input) override;
};
}  // namespace CubbyDNN::Transforms::ImageTransforms

#endif  // CUBBYDNN_IMAGE_TRANSFORMS_HPP
