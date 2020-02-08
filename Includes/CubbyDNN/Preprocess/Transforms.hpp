#ifndef CUBBYDNN_TRANSFORMS_HPP
#define CUBBYDNN_TRANSFORMS_HPP

#include <CubbyDNN/Datas/Image.hpp>

namespace CubbyDNN::Transforms
{
Image Rotation(const Image& origin, double degree);
} // namespace CubbyDNN::Transforms

#endif  // CUBBYDNN_TRANFORMS_HPP
