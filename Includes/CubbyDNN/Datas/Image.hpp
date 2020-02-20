#ifndef CUBBYDNN_IMAGE_HPP
#define CUBBYDNN_IMAGE_HPP

#include <vector>

namespace CubbyDNN
{
class Pixel final
{
 public:
    Pixel() = default;
    Pixel(unsigned char r, unsigned char g, unsigned char b);
    Pixel(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
    explicit Pixel(unsigned char gray);

    unsigned char& R();
    unsigned char R() const;

    unsigned char& G();
    unsigned char G() const;

    unsigned char& B();
    unsigned char B() const;

    unsigned char& A();
    unsigned char A() const;

    unsigned char& Gray();
    unsigned char Gray() const;

    void ToGrayScale();

    bool operator==(const Pixel& other) const;
    bool operator!=(const Pixel& other) const;

 private:
    unsigned char m_r{ 0 }, m_g{ 0 }, m_b{ 0 }, m_a{ 255 };
    bool m_grayScale{ false };
};

class Image final
{
 public:
    Image() = default;
    Image(std::size_t width, std::size_t height, bool hasAlpha = true, bool grayScale = false);

    std::size_t GetWidth() const;
    std::size_t GetHeight() const;
    bool HasAlpha() const;

    Pixel& At(std::size_t x, std::size_t y);
    const Pixel& At(std::size_t x, std::size_t y) const;

 private:
    std::size_t m_width{ 0 }, m_height{ 0 };
    bool m_hasAlpha{ true };
    bool m_grayScale{ false };
    std::vector<Pixel> m_data;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_IMAGE_HPP
