#ifndef CUBBYDNN_TRANSFORMS_HPP
#define CUBBYDNN_TRANSFORMS_HPP

#include <CubbyDNN/Datas/Image.hpp>

namespace CubbyDNN::Transforms
{
Image CenterCrop(const Image& origin, std::size_t size);
Image FlipHorizontal(const Image& origin);
Image FlipVertical(const Image& origin);
Image Rotation(const Image& origin, double degree);
Image GrayScale(const Image& origin);
}  // namespace CubbyDNN::Transforms

#endif  // CUBBYDNN_TRANFORMS_HPP
