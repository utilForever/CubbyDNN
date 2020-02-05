#ifndef CUBBYDNN_IMAGE_HPP
#define CUBBYDNN_IMAGE_HPP

#include <istream>
#include <string>
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

    enum class FileFormat
    {
        BMP,
        UNKNOWN
    };

    Image() = default;
    Image(std::size_t width, std::size_t height);

    std::size_t GetWidth() const;
    std::size_t GetHeight() const;

    static Image Load(const std::string& filename);
    void Save(const std::string& filename, FileFormat format);

    Pixel& At(std::size_t x, std::size_t y);
    const Pixel& At(std::size_t x, std::size_t y) const;

 private:
    static FileFormat checkFileFormat(const std::string& filename);
    static Image loadBMP(const std::string& filename);

    std::size_t m_width{ 0 }, m_height{ 0 };
    std::vector<Pixel> m_data;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_IMAGE_HPP
