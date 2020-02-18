#ifndef CUBBYDNN_IMAGE_HPP
#define CUBBYDNN_IMAGE_HPP

#include <vector>

namespace CubbyDNN
{
class Image final
{
 public:
    struct Pixel
    {
        unsigned char r{ 0 }, g{ 0 }, b{ 0 }, a{ 255 };

        bool operator==(const Pixel& other) const;
        bool operator!=(const Pixel& other) const;
    };

    Image() = default;
    Image(std::size_t width, std::size_t height, bool hasAlpha = true);

    std::size_t GetWidth() const;
    std::size_t GetHeight() const;
    bool HasAlpha() const;

    Pixel& At(std::size_t x, std::size_t y);
    const Pixel& At(std::size_t x, std::size_t y) const;

 private:
    std::size_t m_width{ 0 }, m_height{ 0 };
    bool m_hasAlpha{ true };
    std::vector<Pixel> m_data;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_IMAGE_HPP
