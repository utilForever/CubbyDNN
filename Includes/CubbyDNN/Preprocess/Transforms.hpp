#ifndef CUBBYDNN_TRANSFORMS_HPP
#define CUBBYDNN_TRANSFORMS_HPP

#include <CubbyDNN/Datas/Image.hpp>

namespace CubbyDNN::Transforms
{
template <class InputT, class OutputT>
class Transform
{
 public:
    using InputType = InputT;
    using OutputType = OutputT;

    virtual ~Transform() = default;

    virtual OutputType operator()(const InputType& input) = 0;
};

class CenterCrop : public Transform<Image, Image>
{
 public:
    CenterCrop(std::size_t size);

    Image operator()(const Image& input) override;

 private:
    std::size_t m_cropSize_;
};

class FlipHorizontal : public Transform<Image, Image>
{
 public:
    Image operator()(const Image& input) override;
};

class FlipVertical : public Transform<Image, Image>
{
 public:
    Image operator()(const Image& input) override;
};

class Rotation : public Transform<Image, Image>
{
 public:
    Rotation(double degree);

    Image operator()(const Image& input) override;

 private:
    double m_rotationDegree_;
};

class GrayScale : public Transform<Image, Image>
{
public:
    Image operator()(const Image& input) override;
};
}  // namespace CubbyDNN::Transforms

#endif  // CUBBYDNN_TRANFORMS_HPP
