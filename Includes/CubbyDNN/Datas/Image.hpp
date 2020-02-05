#ifndef CUBBYDNN_IMAGE_HPP
#define CUBBYDNN_IMAGE_HPP

#include <array>
#include <vector>

namespace CubbyDNN
{
class Image final
{
 public:
    struct Pixel
    {
        unsigned char r, g, b;

        bool operator==(const Pixel& other) const;
        bool operator!=(const Pixel& other) const;
    };

    std::size_t GetWidth() const;
    std::size_t GetHeight() const;

    Pixel& At(std::size_t x, std::size_t y);
    const Pixel& At(std::size_t x, std::size_t y) const;

 private:
    std::size_t m_width{ 0 }, m_height{ 0 };
    std::vector<Pixel> m_data;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_IMAGE_HPP
