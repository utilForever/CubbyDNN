#ifndef CUBBYDNN_SHAPE_HPP
#define CUBBYDNN_SHAPE_HPP

#include <vector>

namespace CubbyDNN
{
class Shape
{
 public:
    Shape();

 private:
    std::vector<std::size_t> m_dimension;
};
}  // namespace CubbyDNN

#endif