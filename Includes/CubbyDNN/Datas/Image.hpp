#ifndef CUBBYDNN_IMAGE_HPP
#define CUBBYDNN_IMAGE_HPP

#include <string>
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

    enum class FileFormat
    {
        BMP,
        UNKNOWN
    };

    Image() = default;
    Image(std::size_t width, std::size_t height, bool hasAlpha = true);

    std::size_t GetWidth() const;
    std::size_t GetHeight() const;
    bool HasAlpha() const;

    static Image Load(const std::string& filename);
    void Save(const std::string& filename, FileFormat format) const;

    Pixel& At(std::size_t x, std::size_t y);
    const Pixel& At(std::size_t x, std::size_t y) const;

 private:
    static FileFormat checkFileFormat(const std::string& filename);
    static Image loadBMP(const std::string& filename);

    std::size_t m_width{ 0 }, m_height{ 0 };
    bool m_hasAlpha{ true };
    std::vector<Pixel> m_data;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_IMAGE_HPP
